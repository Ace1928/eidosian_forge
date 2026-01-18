import sys
import os
import builtins
import _sitebuiltins
import io
import stat
def sethelper():
    builtins.help = _sitebuiltins._Helper()