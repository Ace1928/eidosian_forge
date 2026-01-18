import os, time
import re
from xdg.IniFile import IniFile, is_ascii
from xdg.BaseDirectory import xdg_data_dirs
from xdg.Exceptions import NoThemeError, debug
import xdg.Config
def getMaxSize(self, directory):
    value = self.get('MaxSize', type='integer', group=directory)
    if value or value == 0:
        return value
    else:
        return self.getSize(directory)