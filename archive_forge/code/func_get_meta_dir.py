import os
import json
import hashlib
from docker import utils
from docker.constants import IS_WINDOWS_PLATFORM
from docker.constants import DEFAULT_UNIX_SOCKET
from docker.utils.config import find_config_file
def get_meta_dir(name=None):
    meta_dir = os.path.join(get_context_dir(), 'meta')
    if name:
        return os.path.join(meta_dir, get_context_id(name))
    return meta_dir