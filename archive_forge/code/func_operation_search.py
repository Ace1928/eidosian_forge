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
def operation_search(self):
    url = self.vars.restbase + '/search?jql=' + pathname2url(self.vars.jql)
    if self.vars.fields:
        fields = self.vars.fields.keys()
        url = url + '&fields=' + '&fields='.join([pathname2url(f) for f in fields])
    if self.vars.maxresults:
        url = url + '&maxResults=' + str(self.vars.maxresults)
    self.vars.meta = self.get(url)