from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.api import WapiModule
from ..module_utils.api import NIOS_RANGE
from ..module_utils.api import normalize_ib_spec
def convert_range_member_to_struct(module):
    """This function will check the module input to ensure that only one member assignment type is specified at once.
    Member passed in is converted to the correct struct for the API to understand bassed on the member type.
    """
    params = [k for k in module.params.keys() if module.params[k] is not None]
    opts = list(set(params).intersection(['member', 'failover_association', 'ms_server']))
    if len(opts) > 1:
        raise AttributeError("'%s' can not be defined when '%s' is defined!" % (opts[0], opts[1]))
    if 'member' in opts:
        module.params['member'] = {'_struct': 'dhcpmember', 'name': module.params['member']}
        module.params['server_association_type'] = 'MEMBER'
    elif 'failover_association' in opts:
        module.params['server_association_type'] = 'FAILOVER'
    elif 'ms_server' in opts:
        module.params['ms_server'] = {'_struct': 'msdhcpserver', 'ipv4addr': module.params['ms_server']}
        module.params['server_association_type'] = 'MS_SERVER'
    else:
        module.params['server_association_type'] = 'NONE'
    return module