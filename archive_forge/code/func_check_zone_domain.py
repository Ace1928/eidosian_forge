from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import open_url
from ansible.module_utils.basic import json
import ansible.module_utils.six.moves.urllib.error as urllib_error
def check_zone_domain(data, domain):
    """
    Returns true if domain already exists, and false if not.
    """
    exists = False
    if data.status_code in [201, 200]:
        for zone_domain in data.json():
            if zone_domain['domain'] == domain:
                exists = True
    return exists