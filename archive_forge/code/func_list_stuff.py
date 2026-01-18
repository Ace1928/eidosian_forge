from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, respawn_module
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
import errno
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from ansible.module_utils.urls import fetch_file
def list_stuff(self, repoquerybin, stuff):
    qf = '%{name}|%{epoch}|%{version}|%{release}|%{arch}|%{repoid}'
    is_installed_qf = '%{name}|%{epoch}|%{version}|%{release}|%{arch}|installed\n'
    repoq = [repoquerybin, '--show-duplicates', '--plugins', '--quiet']
    if self.disablerepo:
        repoq.extend(['--disablerepo', ','.join(self.disablerepo)])
    if self.enablerepo:
        repoq.extend(['--enablerepo', ','.join(self.enablerepo)])
    if self.installroot != '/':
        repoq.extend(['--installroot', self.installroot])
    if self.conf_file and os.path.exists(self.conf_file):
        repoq += ['-c', self.conf_file]
    if stuff == 'installed':
        return [self.pkg_to_dict(p) for p in sorted(self.is_installed(repoq, '-a', qf=is_installed_qf)) if p.strip()]
    if stuff == 'updates':
        return [self.pkg_to_dict(p) for p in sorted(self.is_update(repoq, '-a', qf=qf)) if p.strip()]
    if stuff == 'available':
        return [self.pkg_to_dict(p) for p in sorted(self.is_available(repoq, '-a', qf=qf)) if p.strip()]
    if stuff == 'repos':
        return [dict(repoid=name, state='enabled') for name in sorted(self.repolist(repoq)) if name.strip()]
    return [self.pkg_to_dict(p) for p in sorted(self.is_installed(repoq, stuff, qf=is_installed_qf) + self.is_available(repoq, stuff, qf=qf)) if p.strip()]