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
def operation_transition(self):
    turl = self.vars.restbase + '/issue/' + self.vars.issue + '/transitions'
    tmeta = self.get(turl)
    target = self.vars.status
    tid = None
    for t in tmeta['transitions']:
        if t['name'] == target:
            tid = t['id']
            break
    else:
        raise ValueError("Failed find valid transition for '%s'" % target)
    fields = dict(self.vars.fields)
    if self.vars.summary is not None:
        fields.update({'summary': self.vars.summary})
    if self.vars.description is not None:
        fields.update({'description': self.vars.description})
    data = {'transition': {'id': tid}, 'fields': fields}
    if self.vars.comment is not None:
        data.update({'update': {'comment': [{'add': {'body': self.vars.comment}}]}})
    url = self.vars.restbase + '/issue/' + self.vars.issue + '/transitions'
    self.vars.meta = self.post(url, data)