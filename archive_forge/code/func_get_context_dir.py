import os
import json
import hashlib
from docker import utils
from docker.constants import IS_WINDOWS_PLATFORM
from docker.constants import DEFAULT_UNIX_SOCKET
from docker.utils.config import find_config_file
def get_context_dir():
    return os.path.join(os.path.dirname(find_config_file() or ''), 'contexts')