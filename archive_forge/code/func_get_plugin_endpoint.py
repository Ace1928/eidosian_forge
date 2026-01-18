from __future__ import absolute_import, division, print_function
import os
import functools
from pprint import pformat
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.parsing.splitter import parse_kv, split_args
from ansible.utils.display import Display
from ansible.module_utils.six import raise_from
from importlib.metadata import version
def get_plugin_endpoint(netbox, plugin, term):
    """
    get_plugin_endpoint(netbox, plugin, term)
        netbox: a predefined pynetbox.api() pointing to a valid instance
                of NetBox
        plugin: a string referencing the plugin name
        term: the term passed to the lookup function upon which the api
              call will be identified
    """
    attr = 'plugins.%s.%s' % (plugin, term)

    def _getattr(netbox, attr):
        return getattr(netbox, attr)
    return functools.reduce(_getattr, [netbox] + attr.split('.'))