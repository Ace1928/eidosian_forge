import os
import sys
from .gui import *
from .app_menus import ListedWindow
def set_autocall(self):
    self.settings['autocall'] = self.autocall.get()