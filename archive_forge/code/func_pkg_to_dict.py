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
def pkg_to_dict(self, pkgstr):
    if pkgstr.strip() and pkgstr.count('|') == 5:
        n, e, v, r, a, repo = pkgstr.split('|')
    else:
        return {'error_parsing': pkgstr}
    d = {'name': n, 'arch': a, 'epoch': e, 'release': r, 'version': v, 'repo': repo, 'envra': '%s:%s-%s-%s.%s' % (e, n, v, r, a)}
    if repo == 'installed':
        d['yumstate'] = 'installed'
    else:
        d['yumstate'] = 'available'
    return d