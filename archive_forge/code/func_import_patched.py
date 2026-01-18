import re
import struct
import sys
import eventlet
from eventlet import patcher
from eventlet.green import _socket_nodns
from eventlet.green import os
from eventlet.green import time
from eventlet.green import select
from eventlet.green import ssl
def import_patched(module_name):
    modules = {'select': select, 'time': time, 'os': os, 'socket': _socket_nodns, 'ssl': ssl}
    return patcher.import_patched(module_name, **modules)