import os
import sys
from .gui import *
from .app_menus import ListedWindow
def current_font_dict(self):
    font_string = self.text_widget.cget('font')
    return Font(font=font_string).actual()