from os.path import dirname as _dirname
from os.path import splitext as _splitext
import pyglet
from pyglet.text import layout, document, caret
@font_name.setter
def font_name(self, font_name):
    self.document.set_style(0, len(self.document.text), {'font_name': font_name})