from __future__ import absolute_import, division, print_function
import base64
import binascii
import json
import mimetypes
import os
import random
import string
import traceback
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper, cause_changes
from ansible.module_utils.six.moves.urllib.request import pathname2url
from ansible.module_utils.common.text.converters import to_text, to_bytes, to_native
from ansible.module_utils.urls import fetch_url
@cause_changes(on_success=True)
def operation_create(self):
    createfields = {'project': {'key': self.vars.project}, 'summary': self.vars.summary, 'issuetype': {'name': self.vars.issuetype}}
    if self.vars.description:
        createfields['description'] = self.vars.description
    if self.vars.fields:
        createfields.update(self.vars.fields)
    data = {'fields': createfields}
    url = self.vars.restbase + '/issue/'
    self.vars.meta = self.post(url, data)