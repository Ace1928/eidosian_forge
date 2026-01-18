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
def expand_pkgspec_from_fnmatches(m, pkgspec, cache):
    new_pkgspec = []
    if pkgspec:
        for pkgspec_pattern in pkgspec:
            if not isinstance(pkgspec_pattern, string_types):
                m.fail_json(msg='Invalid type for package name, expected string but got %s' % type(pkgspec_pattern))
            pkgname_pattern, version_cmp, version = package_split(pkgspec_pattern)
            if frozenset('*?[]!').intersection(pkgname_pattern):
                if ':' not in pkgname_pattern:
                    try:
                        pkg_name_cache = _non_multiarch
                    except NameError:
                        pkg_name_cache = _non_multiarch = [pkg.name for pkg in cache if ':' not in pkg.name]
                else:
                    try:
                        pkg_name_cache = _all_pkg_names
                    except NameError:
                        pkg_name_cache = _all_pkg_names = [pkg.name for pkg in cache]
                matches = fnmatch.filter(pkg_name_cache, pkgname_pattern)
                if not matches:
                    m.fail_json(msg="No package(s) matching '%s' available" % to_text(pkgname_pattern))
                else:
                    new_pkgspec.extend(matches)
            else:
                new_pkgspec.append(pkgspec_pattern)
    return new_pkgspec