from __future__ import absolute_import, division, print_function
import os
import platform
import re
import shlex
import sqlite3
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def parse_package_name(names, pkg_spec, module):
    pkg_spec['package_latest_leftovers'] = []
    for name in names:
        module.debug('parse_package_name(): parsing name: %s' % name)
        version_match = re.search('-[0-9]', name)
        versionless_match = re.search('--', name)
        if version_match and versionless_match:
            module.fail_json(msg='package name both has a version and is version-less: ' + name)
        pkg_spec[name] = {}
        if version_match:
            match = re.search('^(?P<stem>[^%]+)-(?P<version>[0-9][^-]*)(?P<flavor_separator>-)?(?P<flavor>[a-z].*)?(%(?P<branch>.+))?$', name)
            if match:
                pkg_spec[name]['stem'] = match.group('stem')
                pkg_spec[name]['version_separator'] = '-'
                pkg_spec[name]['version'] = match.group('version')
                pkg_spec[name]['flavor_separator'] = match.group('flavor_separator')
                pkg_spec[name]['flavor'] = match.group('flavor')
                pkg_spec[name]['branch'] = match.group('branch')
                pkg_spec[name]['style'] = 'version'
                module.debug('version_match: stem: %(stem)s, version: %(version)s, flavor_separator: %(flavor_separator)s, flavor: %(flavor)s, branch: %(branch)s, style: %(style)s' % pkg_spec[name])
            else:
                module.fail_json(msg='unable to parse package name at version_match: ' + name)
        elif versionless_match:
            match = re.search('^(?P<stem>[^%]+)--(?P<flavor>[a-z].*)?(%(?P<branch>.+))?$', name)
            if match:
                pkg_spec[name]['stem'] = match.group('stem')
                pkg_spec[name]['version_separator'] = '-'
                pkg_spec[name]['version'] = None
                pkg_spec[name]['flavor_separator'] = '-'
                pkg_spec[name]['flavor'] = match.group('flavor')
                pkg_spec[name]['branch'] = match.group('branch')
                pkg_spec[name]['style'] = 'versionless'
                module.debug('versionless_match: stem: %(stem)s, flavor: %(flavor)s, branch: %(branch)s, style: %(style)s' % pkg_spec[name])
            else:
                module.fail_json(msg='unable to parse package name at versionless_match: ' + name)
        else:
            match = re.search('^(?P<stem>[^%]+)(%(?P<branch>.+))?$', name)
            if match:
                pkg_spec[name]['stem'] = match.group('stem')
                pkg_spec[name]['version_separator'] = None
                pkg_spec[name]['version'] = None
                pkg_spec[name]['flavor_separator'] = None
                pkg_spec[name]['flavor'] = None
                pkg_spec[name]['branch'] = match.group('branch')
                pkg_spec[name]['style'] = 'stem'
                module.debug('stem_match: stem: %(stem)s, branch: %(branch)s, style: %(style)s' % pkg_spec[name])
            else:
                module.fail_json(msg='unable to parse package name at else: ' + name)
        if pkg_spec[name]['branch']:
            branch_release = '6.0'
            if LooseVersion(platform.release()) < LooseVersion(branch_release):
                module.fail_json(msg="package name using 'branch' syntax requires at least OpenBSD %s: %s" % (branch_release, name))
        if pkg_spec[name]['flavor']:
            match = re.search('-$', pkg_spec[name]['flavor'])
            if match:
                module.fail_json(msg='trailing dash in flavor: ' + pkg_spec[name]['flavor'])