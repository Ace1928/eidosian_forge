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
def refresh_keys_hint(self):
    layout = self.available_layouts[self.layout]
    layout_cols = layout['cols']
    layout_rows = layout['rows']
    layout_geometry = self.layout_geometry
    mtop, mright, mbottom, mleft = self.margin_hint
    el_hint = 1.0 - mleft - mright
    eh_hint = 1.0 - mtop - mbottom
    ex_hint = 0 + mleft
    ey_hint = 0 + mbottom
    uw_hint = 1.0 / layout_cols * el_hint
    uh_hint = 1.0 / layout_rows * eh_hint
    layout_geometry['U_HINT'] = (uw_hint, uh_hint)
    current_y_hint = ey_hint + eh_hint
    for line_nb in range(1, layout_rows + 1):
        current_y_hint -= uh_hint
        line_name = '%s_%d' % (self.layout_mode, line_nb)
        line_hint = 'LINE_HINT_%d' % line_nb
        layout_geometry[line_hint] = []
        current_x_hint = ex_hint
        for key in layout[line_name]:
            layout_geometry[line_hint].append([(current_x_hint, current_y_hint), (key[3] * uw_hint, uh_hint)])
            current_x_hint += key[3] * uw_hint
    self.layout_geometry = layout_geometry