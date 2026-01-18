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
def add_regen_dependency(self, root: ET.Element) -> None:
    if not self.gen_lite:
        regen_vcxproj = os.path.join(self.environment.get_build_dir(), 'REGEN.vcxproj')
        self.add_project_reference(root, regen_vcxproj, self.environment.coredata.regen_guid)