from __future__ import absolute_import, division, print_function
import traceback
from functools import wraps
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import missing_required_lib
@xcli_wrapper
def connect_ssl(module):
    endpoints = module.params['endpoints']
    username = module.params['username']
    password = module.params['password']
    if not (username and password and endpoints):
        module.fail_json(msg='Username, password or endpoints arguments are missing from the module arguments')
    try:
        return client.XCLIClient.connect_multiendpoint_ssl(username, password, endpoints)
    except errors.CommandFailedConnectionError as e:
        module.fail_json(msg='Connection with Spectrum Accelerate system has failed: {[0]}.'.format(to_native(e)))