import dataclasses
import logging
import os
import re
import traceback
from dataclasses import dataclass
from typing import Iterator, List, Optional, Any, Dict, Tuple, Union
from ray._private import ray_constants
from ray._private.gcs_utils import GcsAioClient
from ray.dashboard.modules.job.common import (
from ray.dashboard.modules.job.pydantic_models import (
from ray.dashboard.modules.job.common import (
from ray.runtime_env import RuntimeEnv
def file_tail_iterator(path: str) -> Iterator[Optional[List[str]]]:
    """Yield lines from a file as it's written.

    Returns lines in batches of up to 10 lines or 20000 characters,
    whichever comes first. If it's a chunk of 20000 characters, then
    the last line that is yielded could be an incomplete line.
    New line characters are kept in the line string.

    Returns None until the file exists or if no new line has been written.
    """
    if not isinstance(path, str):
        raise TypeError(f'path must be a string, got {type(path)}.')
    while not os.path.exists(path):
        logger.debug(f"Path {path} doesn't exist yet.")
        yield None
    with open(path, 'r') as f:
        lines = []
        chunk_char_count = 0
        curr_line = None
        while True:
            if curr_line is None:
                curr_line = f.readline()
            new_chunk_char_count = chunk_char_count + len(curr_line)
            if new_chunk_char_count > MAX_CHUNK_CHAR_LENGTH:
                truncated_line = curr_line[0:MAX_CHUNK_CHAR_LENGTH - chunk_char_count]
                lines.append(truncated_line)
                curr_line = curr_line[MAX_CHUNK_CHAR_LENGTH - chunk_char_count:]
                yield (lines or None)
                lines = []
                chunk_char_count = 0
            elif len(lines) >= 9:
                lines.append(curr_line)
                yield (lines or None)
                lines = []
                chunk_char_count = 0
                curr_line = None
            elif curr_line:
                lines.append(curr_line)
                chunk_char_count = new_chunk_char_count
                curr_line = None
            else:
                yield (lines or None)
                lines = []
                chunk_char_count = 0
                curr_line = None