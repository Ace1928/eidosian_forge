from __future__ import absolute_import, division, print_function
import copy
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
def pubnub_user(module):
    """Create and configure user model if it possible.

    :type module:  AnsibleModule
    :param module: Reference on module which contain module launch
                   information and status report methods.

    :rtype:  User
    :return: Reference on initialized and ready to use user or 'None' in
             case if not all required information has been passed to block.
    """
    user = None
    params = module.params
    if params.get('cache') and params['cache'].get('module_cache'):
        cache = params['cache']['module_cache']
        user = User()
        user.restore(cache=copy.deepcopy(cache['pnm_user']))
    elif params.get('email') and params.get('password'):
        user = User(email=params.get('email'), password=params.get('password'))
    else:
        err_msg = "It looks like not account credentials has been passed or 'cache' field doesn't have result of previous module call."
        module.fail_json(msg='Missing account credentials.', description=err_msg, changed=False)
    return user