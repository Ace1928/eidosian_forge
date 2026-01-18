import os
import sys
from .gui import *
from .app_menus import ListedWindow
def set_ford(self):
    self.settings['cusp_ford_domain'] = self.ford.get()