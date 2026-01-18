from functools import partial
from glob import glob
from distutils.util import convert_path
import distutils.command.build_py as orig
import os
import fnmatch
import textwrap
import io
import distutils.errors
import itertools
import stat
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from ..extern.more_itertools import unique_everseen
from ..warnings import SetuptoolsDeprecationWarning
def analyze_manifest(self):
    self.manifest_files = mf = {}
    if not self.distribution.include_package_data:
        return
    src_dirs = {}
    for package in self.packages or ():
        src_dirs[assert_relative(self.get_package_dir(package))] = package
    if getattr(self, 'existing_egg_info_dir', None) and Path(self.existing_egg_info_dir, 'SOURCES.txt').exists():
        egg_info_dir = self.existing_egg_info_dir
        manifest = Path(egg_info_dir, 'SOURCES.txt')
        files = manifest.read_text(encoding='utf-8').splitlines()
    else:
        self.run_command('egg_info')
        ei_cmd = self.get_finalized_command('egg_info')
        egg_info_dir = ei_cmd.egg_info
        files = ei_cmd.filelist.files
    check = _IncludePackageDataAbuse()
    for path in self._filter_build_files(files, egg_info_dir):
        d, f = os.path.split(assert_relative(path))
        prev = None
        oldf = f
        while d and d != prev and (d not in src_dirs):
            prev = d
            d, df = os.path.split(d)
            f = os.path.join(df, f)
        if d in src_dirs:
            if f == oldf:
                if check.is_module(f):
                    continue
            else:
                importable = check.importable_subpackage(src_dirs[d], f)
                if importable:
                    check.warn(importable)
            mf.setdefault(src_dirs[d], []).append(path)