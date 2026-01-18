import os
import sys
import threading
from . import process
from . import reduction
def assert_spawning(obj):
    if get_spawning_popen() is None:
        raise RuntimeError('%s objects should only be shared between processes through inheritance' % type(obj).__name__)