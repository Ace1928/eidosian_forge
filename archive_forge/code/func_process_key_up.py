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
def process_key_up(self, touch):
    uid = touch.uid
    if self.uid not in touch.ud:
        return
    key_data, key = touch.ud[self.uid]['key']
    displayed_char, internal, special_char, size = key_data
    b_keycode = special_char
    b_modifiers = self._get_modifiers()
    self.dispatch('on_key_up', b_keycode, internal, b_modifiers)
    if special_char == 'capslock':
        uid = -1
    if uid in self.active_keys:
        self.active_keys.pop(uid, None)
        if special_char == 'shift':
            self.have_shift = False
        elif special_char == 'special':
            self.have_special = False
        if special_char == 'capslock' and self.have_capslock:
            self.active_keys[-1] = key
        self.refresh_active_keys_layer()