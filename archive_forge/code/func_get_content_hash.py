import hashlib
import os
import site
import sys
import tarfile
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional
from wasabi import msg
from ..errors import Errors
from ..util import check_spacy_env_vars, download_file, ensure_pathy, get_checksum
from ..util import get_hash, make_tempdir, upload_file
def get_content_hash(loc: Path) -> str:
    return get_checksum(loc)