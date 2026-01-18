import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pip._vendor.packaging.tags import Tag, interpreter_name, interpreter_version
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.exceptions import InvalidWheelFilename
from pip._internal.models.direct_url import DirectUrl
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.utils.temp_dir import TempDirectory, tempdir_kinds
from pip._internal.utils.urls import path_to_url
@staticmethod
def record_download_origin(cache_dir: str, download_info: DirectUrl) -> None:
    origin_path = Path(cache_dir) / ORIGIN_JSON_NAME
    if origin_path.exists():
        try:
            origin = DirectUrl.from_json(origin_path.read_text(encoding='utf-8'))
        except Exception as e:
            logger.warning('Could not read origin file %s in cache entry (%s). Will attempt to overwrite it.', origin_path, e)
        else:
            if origin.url != download_info.url:
                logger.warning('Origin URL %s in cache entry %s does not match download URL %s. This is likely a pip bug or a cache corruption issue. Will overwrite it with the new value.', origin.url, cache_dir, download_info.url)
    origin_path.write_text(download_info.to_json(), encoding='utf-8')