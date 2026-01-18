from __future__ import absolute_import, division, print_function
import re
from fnmatch import fnmatch
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def get_ext_info(self):
    """Get information about existing extensions."""
    res = self.__exec_sql("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'pg_extension')")
    if not res[0]['exists']:
        return True
    query = "SELECT e.extname, e.extversion, n.nspname, c.description FROM pg_catalog.pg_extension AS e LEFT JOIN pg_catalog.pg_namespace AS n ON n.oid = e.extnamespace LEFT JOIN pg_catalog.pg_description AS c ON c.objoid = e.oid AND c.classoid = 'pg_catalog.pg_extension'::pg_catalog.regclass"
    res = self.__exec_sql(query)
    ext_dict = {}
    for i in res:
        ext_ver_raw = i['extversion']
        if re.search('^([0-9]+([\\-]*[0-9]+)?\\.)*[0-9]+([\\-]*[0-9]+)?$', i['extversion']) is None:
            ext_ver = [None, None]
        else:
            ext_ver = i['extversion'].split('.')
            if re.search('-', ext_ver[0]) is not None:
                ext_ver = ext_ver[0].split('-')
            else:
                try:
                    if re.search('-', ext_ver[1]) is not None:
                        ext_ver[1] = ext_ver[1].split('-')[0]
                except IndexError:
                    ext_ver.append(None)
        ext_dict[i['extname']] = dict(extversion=dict(major=int(ext_ver[0]) if ext_ver[0] else None, minor=int(ext_ver[1]) if ext_ver[1] else None, raw=ext_ver_raw), nspname=i['nspname'], description=i['description'])
    return ext_dict