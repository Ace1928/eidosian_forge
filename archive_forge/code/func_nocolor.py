import os
import re
import shutil
import sys
from typing import Dict, Pattern
def nocolor() -> None:
    if sys.platform == 'win32' and colorama is not None:
        colorama.deinit()
    codes.clear()