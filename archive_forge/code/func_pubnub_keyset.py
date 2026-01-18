from __future__ import absolute_import, division, print_function
import copy
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
def pubnub_keyset(module, account, application):
    """Retrieve reference on target keyset from application model.

    NOTE: In case if there is no keyset with specified name, module will
    exit with error.
    :type module:       AnsibleModule
    :param module:      Reference on module which contain module launch
                        information and status report methods.
    :type account:      Account
    :param account:     Reference on PubNub account model which will be
                        used in case of error to export cached data.
    :type application:  Application
    :param application: Reference on PubNub application model from which
                        reference on keyset should be fetched.

    :rtype:  Keyset
    :return: Reference on initialized and ready to use keyset model.
    """
    params = module.params
    keyset = application.keyset(params['keyset'])
    if keyset is None:
        err_fmt = "There is no '{0}' keyset for '{1}' application. Make sure what correct keyset name has been passed. If keyset doesn't exist you can create it on admin.pubnub.com."
        module.fail_json(msg=err_fmt.format(params['keyset'], application.name), changed=account.changed, module_cache=dict(account))
    return keyset