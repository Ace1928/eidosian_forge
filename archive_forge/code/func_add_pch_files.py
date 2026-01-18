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
def add_pch_files(self, pch_sources, lang, inc_cl):
    header = os.path.basename(pch_sources[lang][0])
    pch_file = ET.SubElement(inc_cl, 'PrecompiledHeaderFile')
    pch_file.text = header
    pch_out = ET.SubElement(inc_cl, 'PrecompiledHeaderOutputFile')
    pch_out.text = f'$(IntDir)$(TargetName)-{lang}.pch'
    pch_pdb = ET.SubElement(inc_cl, 'ProgramDataBaseFileName')
    pch_pdb.text = f'$(IntDir)$(TargetName)-{lang}.pdb'
    return header