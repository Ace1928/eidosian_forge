from collections.abc import Iterable
from io import BytesIO
import os
import re
import shutil
import sys
import tempfile
from unittest import TestCase as _TestCase
from fontTools.config import Config
from fontTools.misc.textTools import tobytes
from fontTools.misc.xmlWriter import XMLWriter
def temp_font(self, font_path, file_name):
    self.temp_dir()
    temppath = os.path.join(self.tempdir, file_name)
    shutil.copy2(font_path, temppath)
    return temppath