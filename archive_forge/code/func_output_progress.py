import atexit
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, TypedDict, Union
from urllib.parse import urlparse
from huggingface_hub.constants import REPO_TYPES_URL_PREFIXES, REPOCARD_NAME
from huggingface_hub.repocard import metadata_load, metadata_save
from .hf_api import HfApi, repo_type_and_id_from_hf_id
from .lfs import LFS_MULTIPART_UPLOAD_COMMAND
from .utils import (
from .utils._deprecation import _deprecate_method
def output_progress(stopping_event: threading.Event):
    """
        To be launched as a separate thread with an event meaning it should stop
        the tail.
        """
    pbars: Dict[Tuple[str, str], PbarT] = {}

    def close_pbars():
        for pbar in pbars.values():
            pbar['bar'].update(pbar['bar'].total - pbar['past_bytes'])
            pbar['bar'].refresh()
            pbar['bar'].close()

    def tail_file(filename) -> Iterator[str]:
        """
            Creates a generator to be iterated through, which will return each
            line one by one. Will stop tailing the file if the stopping_event is
            set.
            """
        with open(filename, 'r') as file:
            current_line = ''
            while True:
                if stopping_event.is_set():
                    close_pbars()
                    break
                line_bit = file.readline()
                if line_bit is not None and (not len(line_bit.strip()) == 0):
                    current_line += line_bit
                    if current_line.endswith('\n'):
                        yield current_line
                        current_line = ''
                else:
                    time.sleep(1)
    while not os.path.exists(os.environ['GIT_LFS_PROGRESS']):
        if stopping_event.is_set():
            close_pbars()
            return
        time.sleep(2)
    for line in tail_file(os.environ['GIT_LFS_PROGRESS']):
        try:
            state, file_progress, byte_progress, filename = line.split()
        except ValueError as error:
            raise ValueError(f'Cannot unpack LFS progress line:\n{line}') from error
        description = f'{state.capitalize()} file {filename}'
        current_bytes, total_bytes = byte_progress.split('/')
        current_bytes_int = int(current_bytes)
        total_bytes_int = int(total_bytes)
        pbar = pbars.get((state, filename))
        if pbar is None:
            pbars[state, filename] = {'bar': tqdm(desc=description, initial=current_bytes_int, total=total_bytes_int, unit='B', unit_scale=True, unit_divisor=1024), 'past_bytes': int(current_bytes)}
        else:
            pbar['bar'].update(current_bytes_int - pbar['past_bytes'])
            pbar['past_bytes'] = current_bytes_int