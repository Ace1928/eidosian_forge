from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def hwaddr(value, query='', alias='hwaddr'):
    """Check if string is a HW/MAC address and filter it"""
    query_func_extra_args = {'': ('value',)}
    query_func_map = {'': _empty_hwaddr_query, 'bare': _bare_query, 'bool': _bool_hwaddr_query, 'int': _int_hwaddr_query, 'cisco': _cisco_query, 'eui48': _win_query, 'linux': _linux_query, 'pgsql': _postgresql_query, 'postgresql': _postgresql_query, 'psql': _postgresql_query, 'unix': _unix_query, 'win': _win_query}
    try:
        v = netaddr.EUI(value)
    except Exception:
        v = None
        if query and query != 'bool':
            raise AnsibleFilterError(alias + ': not a hardware address: %s' % value)
    extras = []
    for arg in query_func_extra_args.get(query, tuple()):
        extras.append(locals()[arg])
    try:
        return query_func_map[query](v, *extras)
    except KeyError:
        raise AnsibleFilterError(alias + ': unknown filter type: %s' % query)