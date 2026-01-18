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
def get_env_hash(env: Dict[str, str]) -> str:
    """Construct a hash of the environment variables that will be passed into
    the commands.

    Values in the env dict may be references to the current os.environ, using
    the syntax $ENV_VAR to mean os.environ[ENV_VAR]
    """
    env_vars = {}
    for key, value in env.items():
        if value.startswith('$'):
            env_vars[key] = os.environ.get(value[1:], '')
        else:
            env_vars[key] = value
    return get_hash(env_vars)