from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib_parse import urlparse, parse_qs, urlencode
from urllib.parse import urljoin
from base64 import urlsafe_b64encode
import hashlib
def get_oauthaccesstoken_objectname_from_token(token_name):
    """
      openshift convert the access token to an OAuthAccessToken resource name using the algorithm
      https://github.com/openshift/console/blob/9f352ba49f82ad693a72d0d35709961428b43b93/pkg/server/server.go#L609-L613
    """
    sha256Prefix = 'sha256~'
    content = token_name.strip(sha256Prefix)
    b64encoded = urlsafe_b64encode(hashlib.sha256(content.encode()).digest()).rstrip(b'=')
    return sha256Prefix + b64encoded.decode('utf-8')