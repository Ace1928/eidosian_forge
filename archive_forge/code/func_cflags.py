import sys
import re
import os
from configparser import RawConfigParser
def cflags(self, section='default'):
    val = self.vars.interpolate(self._sections[section]['cflags'])
    return _escape_backslash(val)