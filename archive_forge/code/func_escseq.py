import os
import re
import shutil
import sys
from typing import Dict, Pattern
def escseq(name: str) -> str:
    escape = codes.get(name, '')
    if input_mode and escape and (sys.platform != 'win32'):
        return '\x01' + escape + '\x02'
    else:
        return escape