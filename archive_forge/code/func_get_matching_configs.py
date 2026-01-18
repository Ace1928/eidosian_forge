from ctypes import *
from .base import Display, Screen, ScreenMode, Canvas
from pyglet.libs.darwin.cocoapy import CGDirectDisplayID, quartz, cf
from pyglet.libs.darwin.cocoapy import cfstring_to_string, cfarray_to_list
def get_matching_configs(self, template):
    canvas = CocoaCanvas(self.display, self, None)
    return template.match(canvas)