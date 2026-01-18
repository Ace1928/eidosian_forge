from __future__ import (absolute_import, division, print_function)
import os
import os.path
import socket as pysocket
import struct
from ansible.module_utils.six import PY2
from ansible_collections.community.docker.plugins.module_utils._api.utils import socket as docker_socket
from ansible_collections.community.docker.plugins.module_utils.socket_helper import (
def append_block(stream_id, data):
    if stream_id == docker_socket.STDOUT:
        stdout.append(data)
    elif stream_id == docker_socket.STDERR:
        stderr.append(data)
    else:
        raise ValueError('{0} is not a valid stream ID'.format(stream_id))