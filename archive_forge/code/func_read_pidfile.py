import atexit
import errno
import logging
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from paste.deploy import loadapp, loadserver
from paste.script.command import Command, BadCommand
def read_pidfile(filename):
    if os.path.exists(filename):
        try:
            f = open(filename)
            content = f.read()
            f.close()
            return int(content.strip())
        except (ValueError, IOError):
            return None
    else:
        return None