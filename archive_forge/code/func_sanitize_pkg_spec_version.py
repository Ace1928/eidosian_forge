from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
from ansible_collections.community.general.plugins.module_utils.module_helper import ModuleHelper
def sanitize_pkg_spec_version(self, pkg_spec, version):
    if version is None:
        return pkg_spec
    if pkg_spec.endswith('.tar.gz'):
        self.do_raise(msg="parameter 'version' must not be used when installing from a file")
    if os.path.isdir(pkg_spec):
        self.do_raise(msg="parameter 'version' must not be used when installing from a directory")
    if pkg_spec.endswith('.git'):
        if version.startswith('~'):
            self.do_raise(msg="operator '~' not allowed in version parameter when installing from git repository")
        version = version if version.startswith('@') else '@' + version
    elif version[0] not in ('@', '~'):
        version = '~' + version
    return pkg_spec + version