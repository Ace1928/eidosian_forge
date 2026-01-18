import base64
import hashlib
import hmac
import re
import urllib.parse
from keystoneclient.i18n import _
def canonical_header_str():
    headers_lower = dict(((k.lower().strip(), v.strip()) for k, v in headers.items()))
    user_agent = headers_lower.get('user-agent', '')
    strip_port = re.match('Boto/2\\.[0-9]\\.[0-2]', user_agent)
    header_list = []
    sh_str = auth_param('SignedHeaders')
    for h in sh_str.split(';'):
        if h not in headers_lower:
            continue
        if h == 'host' and strip_port:
            header_list.append('%s:%s' % (h, headers_lower[h].split(':')[0]))
            continue
        header_list.append('%s:%s' % (h, headers_lower[h]))
    return '\n'.join(header_list) + '\n'