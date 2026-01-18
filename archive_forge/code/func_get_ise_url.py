from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import env_fallback
def get_ise_url(hostname, port=None):
    url_result = 'https://{hostname}'.format(hostname=hostname)
    if port:
        url_result = url_result + ':{port}'.format(port=port)
    return url_result