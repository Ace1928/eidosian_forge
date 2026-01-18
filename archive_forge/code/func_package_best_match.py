from __future__ import absolute_import, division, print_function
import warnings
import datetime
import fnmatch
import locale as locale_module
import os
import random
import re
import shutil
import sys
import tempfile
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.six import PY3, string_types
from ansible.module_utils.urls import fetch_file
def package_best_match(pkgname, version_cmp, version, release, cache):
    policy = apt_pkg.Policy(cache)
    policy.read_pinfile(apt_pkg.config.find_file('Dir::Etc::preferences'))
    policy.read_pindir(apt_pkg.config.find_file('Dir::Etc::preferencesparts'))
    if release:
        policy.create_pin('Release', pkgname, release, 990)
    if version_cmp == '=':
        policy.create_pin('Version', pkgname, version, 1001)
    pkg = cache[pkgname]
    pkgver = policy.get_candidate_ver(pkg)
    if not pkgver:
        return None
    if version_cmp == '=' and (not fnmatch.fnmatch(pkgver.ver_str, version)):
        return None
    return pkgver.ver_str