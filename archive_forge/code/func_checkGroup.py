import os, time
import re
from xdg.IniFile import IniFile, is_ascii
from xdg.BaseDirectory import xdg_data_dirs
from xdg.Exceptions import NoThemeError, debug
import xdg.Config
def checkGroup(self, group):
    if not (group == self.defaultGroup or (re.match('^\\[X-', group) and is_ascii(group))):
        self.errors.append('Invalid Group name: %s' % group.encode('ascii', 'replace'))