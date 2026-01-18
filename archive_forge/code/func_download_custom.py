import enum
import io
import os
import posixpath
import tarfile
import warnings
import zipfile
from datetime import datetime
from functools import partial
from itertools import chain
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union
from .. import config
from ..utils import tqdm as hf_tqdm
from ..utils.deprecation_utils import DeprecatedEnum, deprecated
from ..utils.file_utils import (
from ..utils.info_utils import get_size_checksum_dict
from ..utils.logging import get_logger
from ..utils.py_utils import NestedDataStructure, map_nested, size_str
from ..utils.track import TrackedIterable, tracked_str
from .download_config import DownloadConfig
@deprecated('Use `.download`/`.download_and_extract` with `fsspec` URLs instead.')
def download_custom(self, url_or_urls, custom_download):
    """
        Download given urls(s) by calling `custom_download`.

        Args:
            url_or_urls (`str` or `list` or `dict`):
                URL or `list` or `dict` of URLs to download and extract. Each URL is a `str`.
            custom_download (`Callable[src_url, dst_path]`):
                The source URL and destination path. For example
                `tf.io.gfile.copy`, that lets you download from  Google storage.

        Returns:
            downloaded_path(s): `str`, The downloaded paths matching the given input
                `url_or_urls`.

        Example:

        ```py
        >>> downloaded_files = dl_manager.download_custom('s3://my-bucket/data.zip', custom_download_for_my_private_bucket)
        ```
        """
    cache_dir = self.download_config.cache_dir or config.DOWNLOADED_DATASETS_PATH
    max_retries = self.download_config.max_retries

    def url_to_downloaded_path(url):
        return os.path.join(cache_dir, hash_url_to_filename(url))
    downloaded_path_or_paths = map_nested(url_to_downloaded_path, url_or_urls)
    url_or_urls = NestedDataStructure(url_or_urls)
    downloaded_path_or_paths = NestedDataStructure(downloaded_path_or_paths)
    for url, path in zip(url_or_urls.flatten(), downloaded_path_or_paths.flatten()):
        try:
            get_from_cache(url, cache_dir=cache_dir, local_files_only=True, use_etag=False, max_retries=max_retries)
            cached = True
        except FileNotFoundError:
            cached = False
        if not cached or self.download_config.force_download:
            custom_download(url, path)
            get_from_cache(url, cache_dir=cache_dir, local_files_only=True, use_etag=False, max_retries=max_retries)
    self._record_sizes_checksums(url_or_urls, downloaded_path_or_paths)
    return downloaded_path_or_paths.data