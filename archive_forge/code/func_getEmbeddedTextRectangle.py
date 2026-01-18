import os, time
import re
from xdg.IniFile import IniFile, is_ascii
from xdg.BaseDirectory import xdg_data_dirs
from xdg.Exceptions import NoThemeError, debug
import xdg.Config
def getEmbeddedTextRectangle(self):
    """Retrieve the embedded text rectangle from the icon data as a list of
        numbers (x0, y0, x1, y1), if it is specified."""
    return self.get('EmbeddedTextRectangle', type='integer', list=True)