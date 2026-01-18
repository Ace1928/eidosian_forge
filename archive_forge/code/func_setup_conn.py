from __future__ import (absolute_import, division, print_function)
import traceback
def setup_conn(module):
    """
    this function create connection to LXCA
    :param module:
    :return:  lxca connection
    """
    lxca_con = None
    try:
        lxca_con = connect(module.params['auth_url'], module.params['login_user'], module.params['login_password'], 'True')
    except Exception as exception:
        error_msg = '; '.join(exception.args)
        module.fail_json(msg=error_msg, exception=traceback.format_exc())
    return lxca_con