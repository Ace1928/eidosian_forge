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
def add_project_reference(self, root: ET.Element, include: str, projid: str, link_outputs: bool=False) -> None:
    ig = ET.SubElement(root, 'ItemGroup')
    pref = ET.SubElement(ig, 'ProjectReference', Include=include)
    ET.SubElement(pref, 'Project').text = '{%s}' % projid
    if not link_outputs:
        ET.SubElement(pref, 'LinkLibraryDependencies').text = 'false'