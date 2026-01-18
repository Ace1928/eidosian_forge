import errno
import fcntl
import os
import subprocess
import time
from . import Connection, ConnectionException
def lock_path(display):
    return '/tmp/.X%d-lock' % display