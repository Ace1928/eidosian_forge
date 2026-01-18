import os, time
import re
from xdg.IniFile import IniFile, is_ascii
from xdg.BaseDirectory import xdg_data_dirs
from xdg.Exceptions import NoThemeError, debug
import xdg.Config
def getDisplayName(self):
    """Retrieve the display name from the icon data, if one is specified."""
    return self.get('DisplayName', locale=True)