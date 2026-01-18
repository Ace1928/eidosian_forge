import glob
import io
import os
import posixpath
import re
import tarfile
import time
import xml.dom.minidom
import zipfile
from asyncio import TimeoutError
from io import BytesIO
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union
from xml.etree import ElementTree as ET
import fsspec
from aiohttp.client_exceptions import ClientError
from huggingface_hub.utils import EntryNotFoundError
from packaging import version
from .. import config
from ..filesystems import COMPRESSION_FILESYSTEMS
from ..utils.file_utils import (
from ..utils.logging import get_logger
from ..utils.py_utils import map_nested
from .download_config import DownloadConfig
def glob(self, pattern, download_config: Optional[DownloadConfig]=None):
    """Glob function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Args:
            pattern (`str`): Pattern that resulting paths must match.
            download_config : mainly use token or storage_options to support different platforms and auth types.

        Yields:
            [`xPath`]
        """
    posix_path = self.as_posix()
    main_hop, *rest_hops = posix_path.split('::')
    if is_local_path(main_hop):
        yield from Path(main_hop).glob(pattern)
    else:
        if rest_hops:
            urlpath = rest_hops[0]
            urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
            storage_options = {urlpath.split('://')[0]: storage_options}
            posix_path = '::'.join([main_hop, urlpath, *rest_hops[1:]])
        else:
            storage_options = None
        fs, *_ = fsspec.get_fs_token_paths(xjoin(posix_path, pattern), storage_options=storage_options)
        globbed_paths = fs.glob(xjoin(main_hop, pattern))
        for globbed_path in globbed_paths:
            yield type(self)('::'.join([f'{fs.protocol}://{globbed_path}'] + rest_hops))