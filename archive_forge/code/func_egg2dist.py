from __future__ import annotations
import os
import re
import shutil
import stat
import struct
import sys
import sysconfig
import warnings
from email.generator import BytesGenerator, Generator
from email.policy import EmailPolicy
from glob import iglob
from shutil import rmtree
from zipfile import ZIP_DEFLATED, ZIP_STORED
import setuptools
from setuptools import Command
from . import __version__ as wheel_version
from .macosx_libfile import calculate_macosx_platform_tag
from .metadata import pkginfo_to_metadata
from .util import log
from .vendored.packaging import tags
from .vendored.packaging import version as _packaging_version
from .wheelfile import WheelFile
def egg2dist(self, egginfo_path, distinfo_path):
    """Convert an .egg-info directory into a .dist-info directory"""

    def adios(p):
        """Appropriately delete directory, file or link."""
        if os.path.exists(p) and (not os.path.islink(p)) and os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.unlink(p)
    adios(distinfo_path)
    if not os.path.exists(egginfo_path):
        import glob
        pat = os.path.join(os.path.dirname(egginfo_path), '*.egg-info')
        possible = glob.glob(pat)
        err = f'Egg metadata expected at {egginfo_path} but not found'
        if possible:
            alt = os.path.basename(possible[0])
            err += f' ({alt} found - possible misnamed archive file?)'
        raise ValueError(err)
    if os.path.isfile(egginfo_path):
        pkginfo_path = egginfo_path
        pkg_info = pkginfo_to_metadata(egginfo_path, egginfo_path)
        os.mkdir(distinfo_path)
    else:
        pkginfo_path = os.path.join(egginfo_path, 'PKG-INFO')
        pkg_info = pkginfo_to_metadata(egginfo_path, pkginfo_path)
        shutil.copytree(egginfo_path, distinfo_path, ignore=lambda x, y: {'PKG-INFO', 'requires.txt', 'SOURCES.txt', 'not-zip-safe'})
        dependency_links_path = os.path.join(distinfo_path, 'dependency_links.txt')
        with open(dependency_links_path, encoding='utf-8') as dependency_links_file:
            dependency_links = dependency_links_file.read().strip()
        if not dependency_links:
            adios(dependency_links_path)
    pkg_info_path = os.path.join(distinfo_path, 'METADATA')
    serialization_policy = EmailPolicy(utf8=True, mangle_from_=False, max_line_length=0)
    with open(pkg_info_path, 'w', encoding='utf-8') as out:
        Generator(out, policy=serialization_policy).flatten(pkg_info)
    for license_path in self.license_paths:
        filename = os.path.basename(license_path)
        shutil.copy(license_path, os.path.join(distinfo_path, filename))
    adios(egginfo_path)