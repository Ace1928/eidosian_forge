import os
import sys
from .gui import *
from .app_menus import ListedWindow
def set_automagic(self):
    self.settings['automagic'] = self.automagic.get()