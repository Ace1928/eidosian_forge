import os
import sys
from .gui import *
from .app_menus import ListedWindow
def cache_settings(self):
    self.cache.update(self.setting_dict)