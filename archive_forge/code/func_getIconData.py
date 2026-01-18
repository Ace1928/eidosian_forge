import os, time
import re
from xdg.IniFile import IniFile, is_ascii
from xdg.BaseDirectory import xdg_data_dirs
from xdg.Exceptions import NoThemeError, debug
import xdg.Config
def getIconData(path):
    """Retrieve the data from the .icon file corresponding to the given file. If
    there is no .icon file, it returns None.
    
    Example::
    
        getIconData("/usr/share/icons/Tango/scalable/places/folder.svg")
    """
    if os.path.isfile(path):
        icon_file = os.path.splitext(path)[0] + '.icon'
        if os.path.isfile(icon_file):
            data = IconData()
            data.parse(icon_file)
            return data