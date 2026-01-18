import os
import sys
from .gui import *
from .app_menus import ListedWindow
def set_font_sample(self, event=None):
    new_font = self.get_font()
    self.settings['font'] = new_font
    self.sample.tag_config('all', justify=Tk_.CENTER, font=new_font)