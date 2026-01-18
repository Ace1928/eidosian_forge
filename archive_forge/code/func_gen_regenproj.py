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
def gen_regenproj(self):
    if self.gen_lite:
        project_name = 'RECONFIGURE'
        ofname = os.path.join(self.environment.get_build_dir(), 'RECONFIGURE.vcxproj')
        conftype = 'Makefile'
    else:
        project_name = 'REGEN'
        ofname = os.path.join(self.environment.get_build_dir(), 'REGEN.vcxproj')
        conftype = 'Utility'
    guid = self.environment.coredata.regen_guid
    root, type_config = self.create_basic_project(project_name, temp_dir='regen-temp', guid=guid, conftype=conftype)
    if self.gen_lite:
        nmake_base_meson_command, exe_search_paths = Vs2010Backend.get_nmake_base_meson_command_and_exe_search_paths()
        all_configs_prop_group = ET.SubElement(root, 'PropertyGroup')
        multi_config_buildtype_list = coredata.get_genvs_default_buildtype_list()
        _, build_dir_tail = os.path.split(self.src_to_build)
        proj_to_multiconfigured_builds_parent_dir = '..'
        proj_to_src_dir = self.build_to_src
        reconfigure_all_cmd = ''
        for buildtype in multi_config_buildtype_list:
            meson_build_dir_for_buildtype = build_dir_tail[:-2] + buildtype
            proj_to_build_dir_for_buildtype = str(os.path.join(proj_to_multiconfigured_builds_parent_dir, meson_build_dir_for_buildtype))
            reconfigure_all_cmd += f'{nmake_base_meson_command} setup --reconfigure "{proj_to_build_dir_for_buildtype}" "{proj_to_src_dir}"\n'
        ET.SubElement(all_configs_prop_group, 'NMakeBuildCommandLine').text = reconfigure_all_cmd
        ET.SubElement(all_configs_prop_group, 'NMakeReBuildCommandLine').text = reconfigure_all_cmd
        ET.SubElement(all_configs_prop_group, 'NMakeCleanCommandLine').text = ''
        ET.SubElement(all_configs_prop_group, 'ExecutablePath').text = exe_search_paths
    else:
        action = ET.SubElement(root, 'ItemDefinitionGroup')
        midl = ET.SubElement(action, 'Midl')
        ET.SubElement(midl, 'AdditionalIncludeDirectories').text = '%(AdditionalIncludeDirectories)'
        ET.SubElement(midl, 'OutputDirectory').text = '$(IntDir)'
        ET.SubElement(midl, 'HeaderFileName').text = '%(Filename).h'
        ET.SubElement(midl, 'TypeLibraryName').text = '%(Filename).tlb'
        ET.SubElement(midl, 'InterfaceIdentifierFilename').text = '%(Filename)_i.c'
        ET.SubElement(midl, 'ProxyFileName').text = '%(Filename)_p.c'
        regen_command = self.environment.get_build_command() + ['--internal', 'regencheck']
        cmd_templ = 'call %s > NUL\n"%s" "%s"'
        regen_command = cmd_templ % (self.get_vcvars_command(), '" "'.join(regen_command), self.environment.get_scratch_dir())
        self.add_custom_build(root, 'regen', regen_command, deps=self.get_regen_filelist(), outputs=[Vs2010Backend.get_regen_stampfile(self.environment.get_build_dir())], msg='Checking whether solution needs to be regenerated.')
    ET.SubElement(root, 'Import', Project='$(VCTargetsPath)\\Microsoft.Cpp.targets')
    ET.SubElement(root, 'ImportGroup', Label='ExtensionTargets')
    self._prettyprint_vcxproj_xml(ET.ElementTree(root), ofname)