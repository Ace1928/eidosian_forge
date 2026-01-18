from os import environ
from kivy.utils import platform
from kivy.properties import AliasProperty
from kivy.event import EventDispatcher
from kivy.setupconfig import USE_SDL2
from kivy.context import register_context
from kivy._metrics import dpi2px, NUMERIC_FORMATS, dispatch_pixel_scale, \
def get_dpi_rounded(self):
    dpi = self.dpi
    if dpi < 140:
        return 120
    elif dpi < 200:
        return 160
    elif dpi < 280:
        return 240
    return 320