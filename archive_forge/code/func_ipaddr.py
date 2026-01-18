from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.api import WapiModule
from ..module_utils.api import NIOS_HOST_RECORD
from ..module_utils.api import normalize_ib_spec
def ipaddr(module, key, filtered_keys=None):
    """ Transforms the input value into a struct supported by WAPI
    This function will transform the input from the playbook into a struct
    that is valid for WAPI in the form of:
        {
            ipv4addr: <value>,
            mac: <value>
        }
    This function does not validate the values are properly formatted or in
    the acceptable range, that is left to WAPI.
    """
    filtered_keys = filtered_keys or list()
    objects = list()
    for item in module.params[key]:
        objects.append(dict([(k, v) for k, v in iteritems(item) if v is not None and k not in filtered_keys]))
    return objects