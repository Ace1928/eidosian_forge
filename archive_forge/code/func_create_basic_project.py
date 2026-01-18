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
def create_basic_project(self, target_name, *, temp_dir, guid, conftype='Utility', target_ext=None, target_platform=None) -> T.Tuple[ET.Element, ET.Element]:
    root = ET.Element('Project', {'DefaultTargets': 'Build', 'ToolsVersion': '4.0', 'xmlns': 'http://schemas.microsoft.com/developer/msbuild/2003'})
    confitems = ET.SubElement(root, 'ItemGroup', {'Label': 'ProjectConfigurations'})
    if not target_platform:
        target_platform = self.platform
    multi_config_buildtype_list = coredata.get_genvs_default_buildtype_list() if self.gen_lite else [self.buildtype]
    for buildtype in multi_config_buildtype_list:
        prjconf = ET.SubElement(confitems, 'ProjectConfiguration', {'Include': buildtype + '|' + target_platform})
        ET.SubElement(prjconf, 'Configuration').text = buildtype
        ET.SubElement(prjconf, 'Platform').text = target_platform
    globalgroup = ET.SubElement(root, 'PropertyGroup', Label='Globals')
    guidelem = ET.SubElement(globalgroup, 'ProjectGuid')
    guidelem.text = '{%s}' % guid
    kw = ET.SubElement(globalgroup, 'Keyword')
    kw.text = self.platform + 'Proj'
    ET.SubElement(root, 'Import', Project='$(VCTargetsPath)\\Microsoft.Cpp.Default.props')
    type_config = ET.SubElement(root, 'PropertyGroup', Label='Configuration')
    ET.SubElement(type_config, 'ConfigurationType').text = conftype
    if self.platform_toolset:
        ET.SubElement(type_config, 'PlatformToolset').text = self.platform_toolset
    ET.SubElement(root, 'Import', Project='$(VCTargetsPath)\\Microsoft.Cpp.props')
    pname = ET.SubElement(globalgroup, 'ProjectName')
    pname.text = target_name
    if not self.gen_lite:
        ns = ET.SubElement(globalgroup, 'RootNamespace')
        ns.text = target_name
        p = ET.SubElement(globalgroup, 'Platform')
        p.text = target_platform
        if self.windows_target_platform_version:
            ET.SubElement(globalgroup, 'WindowsTargetPlatformVersion').text = self.windows_target_platform_version
        ET.SubElement(globalgroup, 'UseMultiToolTask').text = 'true'
        ET.SubElement(type_config, 'CharacterSet').text = 'MultiByte'
        ET.SubElement(type_config, 'UseOfMfc').text = 'false'
        direlem = ET.SubElement(root, 'PropertyGroup')
        fver = ET.SubElement(direlem, '_ProjectFileVersion')
        fver.text = self.project_file_version
        outdir = ET.SubElement(direlem, 'OutDir')
        outdir.text = '.\\'
        intdir = ET.SubElement(direlem, 'IntDir')
        intdir.text = temp_dir + '\\'
        tname = ET.SubElement(direlem, 'TargetName')
        tname.text = target_name
        if target_ext:
            ET.SubElement(direlem, 'TargetExt').text = target_ext
        ET.SubElement(direlem, 'EmbedManifest').text = 'false'
    return (root, type_config)