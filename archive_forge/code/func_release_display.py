from ctypes import *
from .base import Display, Screen, ScreenMode, Canvas
from pyglet.libs.darwin.cocoapy import CGDirectDisplayID, quartz, cf
from pyglet.libs.darwin.cocoapy import cfstring_to_string, cfarray_to_list
def release_display(self):
    quartz.CGDisplayRelease(self._cg_display_id)