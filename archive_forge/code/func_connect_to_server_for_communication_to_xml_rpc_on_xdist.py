from _pydev_runfiles import pydev_runfiles_xml_rpc
import pickle
import zlib
import base64
import os
from pydevd_file_utils import canonical_normalized_path
import pytest
import sys
import time
from pathlib import Path
def connect_to_server_for_communication_to_xml_rpc_on_xdist():
    global connected
    if connected:
        return
    connected = True
    if is_in_xdist_node():
        port = os.environ.get('PYDEV_PYTEST_SERVER')
        if not port:
            sys.stderr.write('Error: no PYDEV_PYTEST_SERVER environment variable defined.\n')
        else:
            pydev_runfiles_xml_rpc.initialize_server(int(port), daemon=True)