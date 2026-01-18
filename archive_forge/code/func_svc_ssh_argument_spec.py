from __future__ import absolute_import, division, print_function
import json
import logging
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
def svc_ssh_argument_spec():
    """
    Returns argument_spec of options common to ibm_svcinfo_command
    and ibm_svctask_command modules

    :returns: argument_spec
    :rtype: dict
    """
    return dict(clustername=dict(type='str', required=True), username=dict(type='str', required=True), password=dict(type='str', required=True, no_log=True), log_path=dict(type='str'))