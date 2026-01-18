import os
import sys
import threading
from . import process
from . import reduction
def set_spawning_popen(popen):
    _tls.spawning_popen = popen