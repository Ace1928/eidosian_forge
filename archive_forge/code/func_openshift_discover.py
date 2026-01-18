from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib_parse import urlparse, parse_qs, urlencode
from urllib.parse import urljoin
from base64 import urlsafe_b64encode
import hashlib
def openshift_discover(self):
    url = urljoin(self.con_host, '.well-known/oauth-authorization-server')
    ret = requests.get(url, verify=self.con_verify_ca)
    if ret.status_code != 200:
        self.fail_request("Couldn't find OpenShift's OAuth API", method='GET', url=url, reason=ret.reason, status_code=ret.status_code)
    try:
        oauth_info = ret.json()
        self.openshift_auth_endpoint = oauth_info['authorization_endpoint']
        self.openshift_token_endpoint = oauth_info['token_endpoint']
    except Exception:
        self.fail_json(msg='Something went wrong discovering OpenShift OAuth details.', exception=traceback.format_exc())