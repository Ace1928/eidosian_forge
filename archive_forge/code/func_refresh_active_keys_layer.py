from kivy import kivy_data_dir
from kivy.vector import Vector
from kivy.config import Config
from kivy.uix.scatter import Scatter
from kivy.uix.label import Label
from kivy.properties import ObjectProperty, NumericProperty, StringProperty, \
from kivy.logger import Logger
from kivy.graphics import Color, BorderImage, Canvas
from kivy.core.image import Image
from kivy.resources import resource_find
from kivy.clock import Clock
from io import open
from os.path import join, splitext, basename
from os import listdir
from json import loads
def refresh_active_keys_layer(self):
    self.active_keys_layer.clear()
    active_keys = self.active_keys
    layout_geometry = self.layout_geometry
    background = resource_find(self.key_background_down)
    texture = Image(background, mipmap=True).texture
    with self.active_keys_layer:
        Color(*self.key_background_color)
        for line_nb, index in active_keys.values():
            pos, size = layout_geometry['LINE_%d' % line_nb][index]
            BorderImage(texture=texture, pos=pos, size=size, border=self.key_border)