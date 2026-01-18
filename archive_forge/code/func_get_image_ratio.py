from kivy.uix.widget import Widget
from kivy.core.image import Image as CoreImage
from kivy.resources import resource_find
from kivy.properties import (
from kivy.logger import Logger
def get_image_ratio(self):
    if self.texture:
        return self.texture.width / float(self.texture.height)
    return 1.0