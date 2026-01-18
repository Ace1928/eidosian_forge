from __future__ import absolute_import, division, print_function
import json
import os
import random
import mimetypes
from pprint import pformat
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.urls import open_url
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils._text import to_native
import ssl
def ems_log_event(source, server, name='Ansible', id='12345', version=ansible_version, category='Information', event='setup', autosupport='false'):
    ems_log = zapi.NaElement('ems-autosupport-log')
    ems_log.add_new_child('computer-name', name)
    ems_log.add_new_child('event-id', id)
    ems_log.add_new_child('event-source', source)
    ems_log.add_new_child('app-version', version)
    ems_log.add_new_child('category', category)
    ems_log.add_new_child('event-description', event)
    ems_log.add_new_child('log-level', '6')
    ems_log.add_new_child('auto-support', autosupport)
    server.invoke_successfully(ems_log, True)