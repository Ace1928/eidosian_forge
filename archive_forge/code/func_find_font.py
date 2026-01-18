from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
def find_font(self, name, size=12, bold=False, italic=False):
    result = self._get_from_search_cache(name, size, bold, italic)
    if result:
        return result
    search_pattern = self.create_search_pattern()
    search_pattern.name = name
    search_pattern.size = size
    search_pattern.bold = bold
    search_pattern.italic = italic
    result = search_pattern.match()
    self._add_to_search_cache(search_pattern, result)
    search_pattern.dispose()
    return result