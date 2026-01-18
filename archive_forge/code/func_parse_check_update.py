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
@staticmethod
def parse_check_update(check_update_output):
    out = '\n'.join((l for l in check_update_output.splitlines() if l))
    out = re.sub('\\n\\W+(.*)', ' \\1', out)
    updates = {}
    obsoletes = {}
    for line in out.split('\n'):
        line = line.split()
        if '*' in line or len(line) not in [3, 6] or '.' not in line[0]:
            continue
        pkg, version, repo = (line[0], line[1], line[2])
        name, dist = pkg.rsplit('.', 1)
        if name not in updates:
            updates[name] = []
        updates[name].append({'version': version, 'dist': dist, 'repo': repo})
        if len(line) == 6:
            obsolete_pkg, obsolete_version, obsolete_repo = (line[3], line[4], line[5])
            obsolete_name, obsolete_dist = obsolete_pkg.rsplit('.', 1)
            if obsolete_name not in obsoletes:
                obsoletes[obsolete_name] = []
            obsoletes[obsolete_name].append({'version': obsolete_version, 'dist': obsolete_dist, 'repo': obsolete_repo})
    return (updates, obsoletes)