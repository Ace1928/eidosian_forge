import types
from _pydev_bundle import pydev_log
from typing import Tuple, Literal
def is_tracked_frame(self, frame) -> bool:
    for plugin in self.active_plugins:
        if plugin.is_tracked_frame(frame):
            return True
    return False