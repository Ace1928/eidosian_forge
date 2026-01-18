import threading
import sys
from os.path import basename
from _pydev_bundle import pydev_log
from os import scandir
import time
@property
def path_watchers(self):
    return tuple(self._path_watchers)