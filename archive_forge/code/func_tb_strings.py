import sys
import struct
import traceback
import threading
import logging
from paramiko.common import (
from paramiko.config import SSHConfig
def tb_strings():
    return ''.join(traceback.format_exception(*sys.exc_info())).split('\n')