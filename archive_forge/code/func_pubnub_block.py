from __future__ import absolute_import, division, print_function
import copy
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
def pubnub_block(module, account, keyset):
    """Retrieve reference on target keyset from application model.

    NOTE: In case if there is no block with specified name and module
    configured to start/stop it, module will exit with error.
    :type module:   AnsibleModule
    :param module:  Reference on module which contain module launch
                    information and status report methods.
    :type account:  Account
    :param account: Reference on PubNub account model which will be used in
                    case of error to export cached data.
    :type keyset:   Keyset
    :param keyset:  Reference on keyset model from which reference on block
                    should be fetched.

    :rtype:  Block
    :return: Reference on initialized and ready to use keyset model.
    """
    block = None
    params = module.params
    try:
        block = keyset.block(params['name'])
    except (exceptions.KeysetError, exceptions.GeneralPubNubError) as exc:
        exc_msg = _failure_title_from_exception(exc)
        exc_descr = exc.message if hasattr(exc, 'message') else exc.args[0]
        module.fail_json(msg=exc_msg, description=exc_descr, changed=account.changed, module_cache=dict(account))
    if block is None and params['state'] in ['started', 'stopped']:
        block_name = params.get('name')
        module.fail_json(msg="'{0}' block doesn't exists.".format(block_name), changed=account.changed, module_cache=dict(account))
    if block is None and params['state'] == 'present':
        block = Block(name=params.get('name'), description=params.get('description'))
        keyset.add_block(block)
    if block:
        if params.get('changes') and params['changes'].get('name'):
            block.name = params['changes']['name']
        if params.get('description'):
            block.description = params.get('description')
    return block