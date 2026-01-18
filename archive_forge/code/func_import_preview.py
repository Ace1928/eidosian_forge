from __future__ import (absolute_import, division, print_function)
import json
import re
import time
import os
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.parameters import env_fallback
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import config_ipv6
def import_preview(self, import_buffer=None, target=None, share=None, job_wait=False):
    payload = {'ShareParameters': {'Target': target}}
    if import_buffer is not None:
        payload['ImportBuffer'] = import_buffer
    if share is None:
        share = {}
    if share.get('share_ip') is not None:
        payload['ShareParameters']['IPAddress'] = share['share_ip']
    if share.get('share_name') is not None and share.get('share_name'):
        payload['ShareParameters']['ShareName'] = share['share_name']
    if share.get('share_type') is not None:
        payload['ShareParameters']['ShareType'] = share['share_type']
    if share.get('file_name') is not None:
        payload['ShareParameters']['FileName'] = share['file_name']
    if share.get('username') is not None:
        payload['ShareParameters']['Username'] = share['username']
    if share.get('password') is not None:
        payload['ShareParameters']['Password'] = share['password']
    if share.get('ignore_certificate_warning') is not None:
        payload['ShareParameters']['IgnoreCertificateWarning'] = share['ignore_certificate_warning']
    if share.get('proxy_support') is not None:
        payload['ShareParameters']['ProxySupport'] = share['proxy_support']
    if share.get('proxy_type') is not None:
        payload['ShareParameters']['ProxyType'] = share['proxy_type']
    if share.get('proxy_port') is not None:
        payload['ShareParameters']['ProxyPort'] = share['proxy_port']
    if share.get('proxy_server') is not None:
        payload['ShareParameters']['ProxyServer'] = share['proxy_server']
    if share.get('proxy_username') is not None:
        payload['ShareParameters']['ProxyUserName'] = share['proxy_username']
    if share.get('proxy_password') is not None:
        payload['ShareParameters']['ProxyPassword'] = share['proxy_password']
    response = self.invoke_request(IMPORT_PREVIEW, 'POST', data=payload)
    if response.status_code == 202 and job_wait:
        task_uri = response.headers['Location']
        response = self.wait_for_job_complete(task_uri, job_wait=job_wait)
    return response