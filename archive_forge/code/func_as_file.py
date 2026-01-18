import base64
import io
import os
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, BinaryIO, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union
from tqdm.contrib.concurrent import thread_map
from huggingface_hub import get_session
from .constants import ENDPOINT, HF_HUB_ENABLE_HF_TRANSFER
from .file_download import hf_hub_url
from .lfs import UploadInfo, lfs_upload, post_lfs_batch_info
from .utils import (
from .utils import tqdm as hf_tqdm
@contextmanager
def as_file(self, with_tqdm: bool=False) -> Iterator[BinaryIO]:
    """
        A context manager that yields a file-like object allowing to read the underlying
        data behind `path_or_fileobj`.

        Args:
            with_tqdm (`bool`, *optional*, defaults to `False`):
                If True, iterating over the file object will display a progress bar. Only
                works if the file-like object is a path to a file. Pure bytes and buffers
                are not supported.

        Example:

        ```python
        >>> operation = CommitOperationAdd(
        ...        path_in_repo="remote/dir/weights.h5",
        ...        path_or_fileobj="./local/weights.h5",
        ... )
        CommitOperationAdd(path_in_repo='remote/dir/weights.h5', path_or_fileobj='./local/weights.h5')

        >>> with operation.as_file() as file:
        ...     content = file.read()

        >>> with operation.as_file(with_tqdm=True) as file:
        ...     while True:
        ...         data = file.read(1024)
        ...         if not data:
        ...              break
        config.json: 100%|█████████████████████████| 8.19k/8.19k [00:02<00:00, 3.72kB/s]

        >>> with operation.as_file(with_tqdm=True) as file:
        ...     requests.put(..., data=file)
        config.json: 100%|█████████████████████████| 8.19k/8.19k [00:02<00:00, 3.72kB/s]
        ```
        """
    if isinstance(self.path_or_fileobj, str) or isinstance(self.path_or_fileobj, Path):
        if with_tqdm:
            with tqdm_stream_file(self.path_or_fileobj) as file:
                yield file
        else:
            with open(self.path_or_fileobj, 'rb') as file:
                yield file
    elif isinstance(self.path_or_fileobj, bytes):
        yield io.BytesIO(self.path_or_fileobj)
    elif isinstance(self.path_or_fileobj, io.BufferedIOBase):
        prev_pos = self.path_or_fileobj.tell()
        yield self.path_or_fileobj
        self.path_or_fileobj.seek(prev_pos, io.SEEK_SET)