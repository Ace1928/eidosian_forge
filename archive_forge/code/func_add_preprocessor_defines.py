from __future__ import annotations
import copy
import itertools
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
import uuid
import typing as T
from pathlib import Path, PurePath, PureWindowsPath
import re
from collections import Counter
from . import backends
from .. import build
from .. import mlog
from .. import compilers
from .. import mesonlib
from ..mesonlib import (
from ..environment import Environment, build_filename
from .. import coredata
def add_preprocessor_defines(self, lang, parent_node, file_defines):
    defines = []
    for define in file_defines[lang]:
        if define == '%(PreprocessorDefinitions)':
            defines.append(define)
        else:
            defines.append(self.escape_preprocessor_define(define))
    ET.SubElement(parent_node, 'PreprocessorDefinitions').text = ';'.join(defines)