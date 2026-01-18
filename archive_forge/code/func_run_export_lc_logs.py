from __future__ import (absolute_import, division, print_function)
import socket
import json
import copy
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def run_export_lc_logs(idrac, module):
    """
    Export Lifecycle Controller Log to the given file share

    Keyword arguments:
    idrac  -- iDRAC handle
    module -- Ansible module
    """
    lclog_file_name_format = '%ip_%Y%m%d_%H%M%S_LC_Log.log'
    share_username = module.params.get('share_user')
    if share_username is not None and ('@' in share_username or '\\' in share_username):
        myshare = get_user_credentials(module)
    else:
        myshare = file_share_manager.create_share_obj(share_path=module.params['share_name'], creds=UserCredentials(module.params['share_user'], module.params['share_password']), isFolder=True)
    data = socket.getaddrinfo(module.params['idrac_ip'], module.params['idrac_port'])
    if 'AF_INET6' == data[0][0]._name_:
        ip = copy.deepcopy(module.params['idrac_ip'])
        lclog_file_name_format = '{ip}_%Y%m%d_%H%M%S_LC_Log.log'.format(ip=ip.replace(':', '.').replace('..', '.'))
    lc_log_file = myshare.new_file(lclog_file_name_format)
    job_wait = module.params['job_wait']
    msg = idrac.log_mgr.lclog_export(lc_log_file, job_wait)
    return msg