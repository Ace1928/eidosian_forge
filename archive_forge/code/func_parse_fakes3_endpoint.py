import string
from urllib.parse import urlparse
from ansible.module_utils.basic import to_text
def parse_fakes3_endpoint(url):
    fakes3 = urlparse(url)
    protocol = 'http'
    port = fakes3.port or 80
    if fakes3.scheme == 'fakes3s':
        protocol = 'https'
        port = fakes3.port or 443
    endpoint_url = f'{protocol}://{fakes3.hostname}:{to_text(port)}'
    use_ssl = bool(fakes3.scheme == 'fakes3s')
    return {'endpoint': endpoint_url, 'use_ssl': use_ssl}