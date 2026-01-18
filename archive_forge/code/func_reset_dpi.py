from os import environ
from kivy.utils import platform
from kivy.properties import AliasProperty
from kivy.event import EventDispatcher
from kivy.setupconfig import USE_SDL2
from kivy.context import register_context
from kivy._metrics import dpi2px, NUMERIC_FORMATS, dispatch_pixel_scale, \
def reset_dpi(self, *args):
    """Resets the dpi (and possibly density) to the platform values,
        overwriting any manually set values.
        """
    self.dpi = self.get_dpi(force_recompute=True)