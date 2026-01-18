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
def generate_custom_generator_commands(self, target, parent_node):
    generator_output_files = []
    custom_target_include_dirs = []
    custom_target_output_files = []
    for genlist in target.get_generated_sources():
        self.generate_genlist_for_target(genlist, target, parent_node, generator_output_files, custom_target_include_dirs, custom_target_output_files)
    return (generator_output_files, custom_target_output_files, custom_target_include_dirs)