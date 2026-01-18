from __future__ import (absolute_import, division, print_function)
import os
import os.path
import socket as pysocket
import struct
from ansible.module_utils.six import PY2
from ansible_collections.community.docker.plugins.module_utils._api.utils import socket as docker_socket
from ansible_collections.community.docker.plugins.module_utils.socket_helper import (
def set_block_done_callback(self, block_done_callback):
    self._block_done_callback = block_done_callback
    if self._block_done_callback is not None:
        while self._block_buffer:
            elt = self._block_buffer.remove(0)
            self._block_done_callback(*elt)