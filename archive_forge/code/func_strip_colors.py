import os
import re
import shutil
import sys
from typing import Dict, Pattern
def strip_colors(s: str) -> str:
    return re.compile('\x1b.*?m').sub('', s)