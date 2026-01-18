from kivy.uix.widget import Widget
from kivy.core.image import Image as CoreImage
from kivy.resources import resource_find
from kivy.properties import (
from kivy.logger import Logger
def on_anim_delay(self, instance, value):
    if self._coreimage is None:
        return
    self._coreimage.anim_delay = value
    if value < 0:
        self._coreimage.anim_reset(False)