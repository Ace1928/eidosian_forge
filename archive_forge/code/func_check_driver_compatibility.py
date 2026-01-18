from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib  # pylint: disable=unused-import:
from ansible.module_utils.six.moves import configparser
from ansible.module_utils._text import to_native
import traceback
import os
import ssl as ssl_lib
def check_driver_compatibility(module, client, srv_version):
    try:
        driver_version = PyMongoVersion
        check_compatibility(module, srv_version, driver_version)
    except Exception as excep:
        module.fail_json(msg='Unable to check driver compatibility: %s' % to_native(excep))