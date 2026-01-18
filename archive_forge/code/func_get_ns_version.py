from __future__ import absolute_import, division, print_function
import json
import re
import sys
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils._text import to_native
def get_ns_version(client):
    from nssrc.com.citrix.netscaler.nitro.resource.config.ns.nsversion import nsversion
    result = nsversion.get(client)
    m = re.match('^.*NS(\\d+)\\.(\\d+).*$', result[0].version)
    if m is None:
        return None
    else:
        return (int(m.group(1)), int(m.group(2)))