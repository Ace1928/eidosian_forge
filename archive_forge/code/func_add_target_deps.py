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
def add_target_deps(self, root: ET.Element, target):
    target_dict = {target.get_id(): target}
    for dep in self.get_target_deps(target_dict).values():
        if dep.get_id() in self.handled_target_deps[target.get_id()]:
            continue
        relpath = self.get_target_dir_relative_to(dep, target)
        vcxproj = os.path.join(relpath, dep.get_id() + '.vcxproj')
        tid = self.environment.coredata.target_guids[dep.get_id()]
        self.add_project_reference(root, vcxproj, tid)