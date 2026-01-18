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
def get_key_at_pos(self, x, y):
    w, h = self.size
    x_hint = x / w
    layout_geometry = self.layout_geometry
    layout = self.available_layouts[self.layout]
    layout_rows = layout['rows']
    mtop, mright, mbottom, mleft = self.margin_hint
    e_height = h - (mbottom + mtop) * h
    line_height = e_height / layout_rows
    y = y - mbottom * h
    line_nb = layout_rows - int(y / line_height)
    if line_nb > layout_rows:
        line_nb = layout_rows
    if line_nb < 1:
        line_nb = 1
    key_index = ''
    current_key_index = 0
    for key in layout_geometry['LINE_HINT_%d' % line_nb]:
        if x_hint >= key[0][0] and x_hint < key[0][0] + key[1][0]:
            key_index = current_key_index
            break
        else:
            current_key_index += 1
    if key_index == '':
        return None
    key = layout['%s_%d' % (self.layout_mode, line_nb)][key_index]
    return [key, (line_nb, key_index)]