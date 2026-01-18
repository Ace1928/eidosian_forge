import os
import platform
import subprocess
import sys
from ._version import version as __version__
def ninja():
    raise SystemExit(_program('ninja', sys.argv[1:]))