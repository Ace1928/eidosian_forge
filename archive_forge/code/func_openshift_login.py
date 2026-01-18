from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib_parse import urlparse, parse_qs, urlencode
from urllib.parse import urljoin
from base64 import urlsafe_b64encode
import hashlib
def openshift_login(self):
    os_oauth = OAuth2Session(client_id='openshift-challenging-client')
    authorization_url, state = os_oauth.authorization_url(self.openshift_auth_endpoint, state='1', code_challenge_method='S256')
    auth_headers = make_headers(basic_auth='{0}:{1}'.format(self.auth_username, self.auth_password))
    ret = os_oauth.get(authorization_url, headers={'X-Csrf-Token': state, 'authorization': auth_headers.get('authorization')}, verify=self.con_verify_ca, allow_redirects=False)
    if ret.status_code != 302:
        self.fail_request('Authorization failed.', method='GET', url=authorization_url, reason=ret.reason, status_code=ret.status_code)
    qwargs = {}
    for k, v in parse_qs(urlparse(ret.headers['Location']).query).items():
        qwargs[k] = v[0]
    qwargs['grant_type'] = 'authorization_code'
    ret = os_oauth.post(self.openshift_token_endpoint, headers={'Accept': 'application/json', 'Content-Type': 'application/x-www-form-urlencoded', 'Authorization': 'Basic b3BlbnNoaWZ0LWNoYWxsZW5naW5nLWNsaWVudDo='}, data=urlencode(qwargs), verify=self.con_verify_ca)
    if ret.status_code != 200:
        self.fail_request('Failed to obtain an authorization token.', method='POST', url=self.openshift_token_endpoint, reason=ret.reason, status_code=ret.status_code)
    return ret.json()['access_token']