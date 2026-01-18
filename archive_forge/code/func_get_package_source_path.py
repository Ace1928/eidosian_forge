from __future__ import absolute_import, division, print_function
import os
import platform
import re
import shlex
import sqlite3
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_package_source_path(name, pkg_spec, module):
    pkg_spec[name]['subpackage'] = None
    if pkg_spec[name]['stem'] == 'sqlports':
        return 'databases/sqlports'
    else:
        sqlports_db_file = '/usr/local/share/sqlports'
        if not os.path.isfile(sqlports_db_file):
            module.fail_json(msg="sqlports file '%s' is missing" % sqlports_db_file)
        conn = sqlite3.connect(sqlports_db_file)
        first_part_of_query = 'SELECT fullpkgpath, fullpkgname FROM ports WHERE fullpkgname'
        query = first_part_of_query + ' = ?'
        module.debug('package_package_source_path(): exact query: %s' % query)
        cursor = conn.execute(query, (name,))
        results = cursor.fetchall()
        if len(results) < 1:
            looking_for = pkg_spec[name]['stem'] + (pkg_spec[name]['version_separator'] or '-') + (pkg_spec[name]['version'] or '%')
            query = first_part_of_query + ' LIKE ?'
            if pkg_spec[name]['flavor']:
                looking_for += pkg_spec[name]['flavor_separator'] + pkg_spec[name]['flavor']
                module.debug('package_package_source_path(): fuzzy flavor query: %s' % query)
                cursor = conn.execute(query, (looking_for,))
            elif pkg_spec[name]['style'] == 'versionless':
                query += ' AND fullpkgname NOT LIKE ?'
                module.debug('package_package_source_path(): fuzzy versionless query: %s' % query)
                cursor = conn.execute(query, (looking_for, '%s-%%' % looking_for))
            else:
                module.debug('package_package_source_path(): fuzzy query: %s' % query)
                cursor = conn.execute(query, (looking_for,))
            results = cursor.fetchall()
        conn.close()
        if len(results) < 1:
            module.fail_json(msg="could not find a port by the name '%s'" % name)
        if len(results) > 1:
            matches = map(lambda x: x[1], results)
            module.fail_json(msg='too many matches, unsure which to build: %s' % ' OR '.join(matches))
        fullpkgpath = results[0][0]
        parts = fullpkgpath.split(',')
        if len(parts) > 1 and parts[1][0] == '-':
            pkg_spec[name]['subpackage'] = parts[1]
        return parts[0]