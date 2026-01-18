import os
import json
import hashlib
from docker import utils
from docker.constants import IS_WINDOWS_PLATFORM
from docker.constants import DEFAULT_UNIX_SOCKET
from docker.utils.config import find_config_file
def get_current_context_name():
    name = 'default'
    docker_cfg_path = find_config_file()
    if docker_cfg_path:
        try:
            with open(docker_cfg_path) as f:
                name = json.load(f).get('currentContext', 'default')
        except Exception:
            return 'default'
    return name