import re
import os
from ast import literal_eval
from functools import partial
from copy import copy
from kivy import kivy_data_dir
from kivy.config import Config
from kivy.utils import platform
from kivy.graphics.texture import Texture
from kivy.core import core_select_lib
from kivy.core.text.text_layout import layout_text, LayoutWord
from kivy.resources import resource_find, resource_add_path
from kivy.compat import PY2
from kivy.setupconfig import USE_SDL2, USE_PANGOFT2
from kivy.logger import Logger
def resolve_font_name(self):
    options = self.options
    fontname = options['font_name']
    fonts = self._fonts
    fontscache = self._fonts_cache
    if self._font_family_support and options['font_family']:
        options['font_name_r'] = None
        return
    if fontname in fonts:
        italic = int(options['italic'])
        if options['bold']:
            bold = FONT_BOLD
        else:
            bold = FONT_REGULAR
        options['font_name_r'] = fonts[fontname][italic | bold]
    elif fontname in fontscache:
        options['font_name_r'] = fontscache[fontname]
    else:
        filename = resource_find(fontname)
        if not filename and (not fontname.endswith('.ttf')):
            fontname = '{}.ttf'.format(fontname)
            filename = resource_find(fontname)
        if filename is None:
            filename = pep8_fn = os.path.join(kivy_data_dir, fontname)
            if not os.path.exists(pep8_fn) or not os.path.isfile(pep8_fn):
                raise IOError('Label: File %r not found' % fontname)
        fontscache[fontname] = filename
        options['font_name_r'] = filename