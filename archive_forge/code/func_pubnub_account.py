from __future__ import absolute_import, division, print_function
import copy
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
def pubnub_account(module, user):
    """Create and configure account if it is possible.

    :type module:  AnsibleModule
    :param module: Reference on module which contain module launch
                   information and status report methods.
    :type user:    User
    :param user:   Reference on authorized user for which one of accounts
                   should be used during manipulations with block.

    :rtype:  Account
    :return: Reference on initialized and ready to use account or 'None' in
             case if not all required information has been passed to block.
    """
    params = module.params
    if params.get('account'):
        account_name = params.get('account')
        account = user.account(name=params.get('account'))
        if account is None:
            err_frmt = "It looks like there is no '{0}' account for authorized user. Please make sure what correct name has been passed during module configuration."
            module.fail_json(msg='Missing account.', description=err_frmt.format(account_name), changed=False)
    else:
        account = user.accounts()[0]
    return account