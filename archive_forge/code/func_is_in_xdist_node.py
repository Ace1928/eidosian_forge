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
def is_in_xdist_node():
    main_pid = os.environ.get('PYDEV_MAIN_PID')
    if main_pid and main_pid != str(os.getpid()):
        return True
    return False