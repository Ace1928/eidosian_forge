import os, time
import re
from xdg.IniFile import IniFile, is_ascii
from xdg.BaseDirectory import xdg_data_dirs
from xdg.Exceptions import NoThemeError, debug
import xdg.Config
def checkExtras(self):
    if self.fileExtension != '.icon':
        self.warnings.append('Unknown File extension')