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
def get_ephem_path_for_link(self, link: Link) -> str:
    return self._ephem_cache.get_path_for_link(link)