from __future__ import absolute_import, division, print_function
import json
import os
import sys
import uuid
import random
import re
import socket
from datetime import datetime
from ssl import SSLError
from http.client import RemoteDisconnected
from time import time
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import (
from .constants import (
from .version import CURRENT_COLL_VERSION
def build_telemetry(self):
    platform = self.get_platform()
    self.in_ci, ci_name = in_cicd()
    python_version = sys.version.split(' ', maxsplit=1)[0]
    return [{'CollectionName': '{0}'.format(self.coll_name), 'CollectionVersion': CURRENT_COLL_VERSION, 'CollectionModuleName': self.module_name, 'f5Platform': platform, 'f5SoftwareVersion': self.version if self.version else 'none', 'ControllerAnsibleVersion': self.ansible_version, 'ControllerPythonVersion': python_version, 'ControllerAsDocker': self.docker, 'DockerHostname': socket.gethostname() if self.docker else 'none', 'RunningInCiEnv': self.in_ci, 'CiEnvName': ci_name if self.in_ci else 'none'}]