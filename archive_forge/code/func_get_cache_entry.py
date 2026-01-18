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
def get_cache_entry(self, link: Link, package_name: Optional[str], supported_tags: List[Tag]) -> Optional[CacheEntry]:
    """Returns a CacheEntry with a link to a cached item if it exists or
        None. The cache entry indicates if the item was found in the persistent
        or ephemeral cache.
        """
    retval = self._wheel_cache.get(link=link, package_name=package_name, supported_tags=supported_tags)
    if retval is not link:
        return CacheEntry(retval, persistent=True)
    retval = self._ephem_cache.get(link=link, package_name=package_name, supported_tags=supported_tags)
    if retval is not link:
        return CacheEntry(retval, persistent=False)
    return None