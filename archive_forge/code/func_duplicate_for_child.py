import os
import msvcrt
import signal
import sys
import _winapi
from .context import reduction, get_spawning_popen, set_spawning_popen
from . import spawn
from . import util
def duplicate_for_child(self, handle):
    assert self is get_spawning_popen()
    return reduction.duplicate(handle, self.sentinel)