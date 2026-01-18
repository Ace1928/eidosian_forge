from __future__ import absolute_import, division, print_function
import json
import os
import random
import string
import gzip
from io import BytesIO
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import text_type
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_health_resource(self, subsystem, uri, health, expanded):
    status = 'Status'
    if expanded:
        d = expanded
    else:
        r = self.get_request(self.root_uri + uri)
        if r.get('ret'):
            d = r.get('data')
        else:
            return
    if 'Members' in d:
        for m in d.get('Members'):
            u = m.get('@odata.id')
            r = self.get_request(self.root_uri + u)
            if r.get('ret'):
                p = r.get('data')
                if p:
                    e = {self.to_singular(subsystem.lower()) + '_uri': u, status: p.get(status, 'Status not available')}
                    health[subsystem].append(e)
    else:
        e = {self.to_singular(subsystem.lower()) + '_uri': uri, status: d.get(status, 'Status not available')}
        health[subsystem].append(e)