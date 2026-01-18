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
def package_status(m, pkgname, version_cmp, version, default_release, cache, state):
    """
    :return: A tuple of (installed, installed_version, version_installable, has_files). *installed* indicates whether
    the package (regardless of version) is installed. *installed_version* indicates whether the installed package
    matches the provided version criteria. *version_installable* provides the latest matching version that can be
    installed. In the case of virtual packages where we can't determine an applicable match, True is returned.
    *has_files* indicates whether the package has files on the filesystem (even if not installed, meaning a purge is
    required).
    """
    try:
        pkg = cache[pkgname]
        ll_pkg = cache._cache[pkgname]
    except KeyError:
        if state == 'install':
            try:
                provided_packages = cache.get_providing_packages(pkgname)
                if provided_packages:
                    if cache.is_virtual_package(pkgname) and len(provided_packages) == 1:
                        package = provided_packages[0]
                        installed, installed_version, version_installable, has_files = package_status(m, package.name, version_cmp, version, default_release, cache, state='install')
                        if installed:
                            return (installed, installed_version, version_installable, has_files)
                    return (False, False, True, False)
                m.fail_json(msg="No package matching '%s' is available" % pkgname)
            except AttributeError:
                return (False, False, True, False)
        else:
            return (False, False, None, False)
    try:
        has_files = len(pkg.installed_files) > 0
    except UnicodeDecodeError:
        has_files = True
    except AttributeError:
        has_files = False
    try:
        package_is_installed = ll_pkg.current_state == apt_pkg.CURSTATE_INSTALLED
    except AttributeError:
        try:
            package_is_installed = pkg.is_installed
        except AttributeError:
            package_is_installed = pkg.isInstalled
    version_best = package_best_match(pkgname, version_cmp, version, default_release, cache._cache)
    version_is_installed = False
    version_installable = None
    if package_is_installed:
        try:
            installed_version = pkg.installed.version
        except AttributeError:
            installed_version = pkg.installedVersion
        if version_cmp == '=':
            version_is_installed = fnmatch.fnmatch(installed_version, version)
            if version_best and installed_version != version_best and fnmatch.fnmatch(version_best, version):
                version_installable = version_best
        elif version_cmp == '>=':
            version_is_installed = apt_pkg.version_compare(installed_version, version) >= 0
            if version_best and installed_version != version_best and (apt_pkg.version_compare(version_best, version) >= 0):
                version_installable = version_best
        else:
            version_is_installed = True
            if version_best and installed_version != version_best:
                version_installable = version_best
    else:
        version_installable = version_best
    return (package_is_installed, version_is_installed, version_installable, has_files)