import sys
import os
from pathlib import Path
import io
def open_latin1(filename, mode='r'):
    return open(filename, mode=mode, encoding='iso-8859-1')