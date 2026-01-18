import os
import re
import sys
import shlex
import copy
from distutils.command import build_ext
from distutils.dep_util import newer_group, newer
from distutils.util import get_platform
from distutils.errors import DistutilsError, DistutilsSetupError
from numpy.distutils import log
from numpy.distutils.misc_util import (
from numpy.distutils.from_template import process_file as process_f_file
from numpy.distutils.conv_template import process_file as process_c_file
def build_npy_pkg_config(self):
    log.info('build_src: building npy-pkg config files')
    install_cmd = copy.copy(get_cmd('install'))
    if not install_cmd.finalized == 1:
        install_cmd.finalize_options()
    build_npkg = False
    if self.inplace == 1:
        top_prefix = '.'
        build_npkg = True
    elif hasattr(install_cmd, 'install_libbase'):
        top_prefix = install_cmd.install_libbase
        build_npkg = True
    if build_npkg:
        for pkg, infos in self.distribution.installed_pkg_config.items():
            pkg_path = self.distribution.package_dir[pkg]
            prefix = os.path.join(os.path.abspath(top_prefix), pkg_path)
            d = {'prefix': prefix}
            for info in infos:
                install_dir, generated = self._build_npy_pkg_config(info, d)
                self.distribution.data_files.append((install_dir, [generated]))