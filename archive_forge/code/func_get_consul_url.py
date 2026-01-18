from __future__ import absolute_import, division, print_function
import copy
import json
from ansible.module_utils.six.moves.urllib import error as urllib_error
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import open_url
def get_consul_url(configuration):
    return '%s://%s:%s/v1' % (configuration.scheme, configuration.host, configuration.port)