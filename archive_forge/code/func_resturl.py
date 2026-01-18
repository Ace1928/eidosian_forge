from __future__ import absolute_import, division, print_function
import json
import logging
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
@property
def resturl(self):
    if self.domain:
        hostname = '%s.%s' % (self.clustername, self.domain)
    else:
        hostname = self.clustername
    return getattr(self, '_resturl', None) or '{protocol}://{host}:{port}/rest'.format(protocol=self.protocol, host=hostname, port=self.port)