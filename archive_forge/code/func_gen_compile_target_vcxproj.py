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
def gen_compile_target_vcxproj(self, target: build.CompileTarget, ofname: str, guid: str) -> None:
    if target.for_machine is MachineChoice.BUILD:
        platform = self.build_platform
    else:
        platform = self.platform
    root, type_config = self.create_basic_project(target.name, temp_dir=target.get_id(), guid=guid, target_platform=platform)
    ET.SubElement(root, 'Import', Project='$(VCTargetsPath)\\Microsoft.Cpp.targets')
    target.generated = [self.compile_target_to_generator(target)]
    target.sources = []
    self.generate_custom_generator_commands(target, root)
    self.add_regen_dependency(root)
    self.add_target_deps(root, target)
    self._prettyprint_vcxproj_xml(ET.ElementTree(root), ofname)