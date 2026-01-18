import os
import json
import hashlib
from docker import utils
from docker.constants import IS_WINDOWS_PLATFORM
from docker.constants import DEFAULT_UNIX_SOCKET
from docker.utils.config import find_config_file
def get_tls_dir(name=None, endpoint=''):
    context_dir = get_context_dir()
    if name:
        return os.path.join(context_dir, 'tls', get_context_id(name), endpoint)
    return os.path.join(context_dir, 'tls')