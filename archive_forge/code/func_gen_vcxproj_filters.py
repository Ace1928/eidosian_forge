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
def gen_vcxproj_filters(self, target, ofname):
    root = ET.Element('Project', {'ToolsVersion': '4.0', 'xmlns': 'http://schemas.microsoft.com/developer/msbuild/2003'})
    filter_folders = ET.SubElement(root, 'ItemGroup')
    filter_items = ET.SubElement(root, 'ItemGroup')
    mlog.debug(f'Generating vcxproj filters {target.name}.')

    def relative_to_defined_in(file):
        return os.path.dirname(self.relpath(PureWindowsPath(file.subdir, file.fname), self.get_target_dir(target)))
    found_folders_to_filter = {}
    all_files = target.sources + target.extra_files
    for i in all_files:
        if not os.path.isabs(i.fname):
            dirname = relative_to_defined_in(i)
            if dirname:
                found_folders_to_filter[dirname] = ''
    for folder in found_folders_to_filter:
        dirname = folder
        filter = ''
        while dirname:
            basename = os.path.basename(dirname)
            if filter == '':
                filter = basename
            else:
                filter = basename + ('\\' if dirname in found_folders_to_filter else '/') + filter
            dirname = os.path.dirname(dirname)
        if filter != '':
            found_folders_to_filter[folder] = filter
            filter_element = ET.SubElement(filter_folders, 'Filter', {'Include': filter})
            uuid_element = ET.SubElement(filter_element, 'UniqueIdentifier')
            uuid_element.text = '{' + str(uuid.uuid4()).upper() + '}'
    sources, headers, objects, _ = self.split_sources(all_files)
    down = self.target_to_build_root(target)

    def add_element(type_name, elements):
        for i in elements:
            if not os.path.isabs(i.fname):
                dirname = relative_to_defined_in(i)
                if dirname and dirname in found_folders_to_filter:
                    relpath = os.path.join(down, i.rel_to_builddir(self.build_to_src))
                    target_element = ET.SubElement(filter_items, type_name, {'Include': relpath})
                    filter_element = ET.SubElement(target_element, 'Filter')
                    filter_element.text = found_folders_to_filter[dirname]
    add_element('ClCompile', sources)
    add_element('ClInclude', headers)
    add_element('Object', objects)
    self._prettyprint_vcxproj_xml(ET.ElementTree(root), ofname + '.filters')