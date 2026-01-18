from __future__ import absolute_import, division, print_function
import os
import re
import logging
import sys
from copy import deepcopy
from ansible.module_utils.basic import env_fallback
def avi_common_argument_spec():
    """
    Returns common arguments for all Avi modules
    :return: dict
    """
    credentials_spec = dict(controller=dict(fallback=(env_fallback, ['AVI_CONTROLLER'])), username=dict(fallback=(env_fallback, ['AVI_USERNAME'])), password=dict(fallback=(env_fallback, ['AVI_PASSWORD']), no_log=True), api_version=dict(default='16.4.4', type='str'), tenant=dict(default='admin'), tenant_uuid=dict(default='', type='str'), port=dict(type='int'), timeout=dict(default=300, type='int'), token=dict(default='', type='str', no_log=True), session_id=dict(default='', type='str', no_log=True), csrftoken=dict(default='', type='str', no_log=True))
    return dict(controller=dict(fallback=(env_fallback, ['AVI_CONTROLLER'])), username=dict(fallback=(env_fallback, ['AVI_USERNAME'])), password=dict(fallback=(env_fallback, ['AVI_PASSWORD']), no_log=True), tenant=dict(default='admin'), tenant_uuid=dict(default=''), api_version=dict(default='16.4.4', type='str'), avi_credentials=dict(default=None, type='dict', options=credentials_spec), api_context=dict(type='dict'), avi_disable_session_cache_as_fact=dict(default=False, type='bool'))