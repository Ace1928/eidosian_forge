from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import integer_types, string_types
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
import traceback
def gitlab_authentication(module):
    ensure_gitlab_package(module)
    gitlab_url = module.params['api_url']
    validate_certs = module.params['validate_certs']
    ca_path = module.params['ca_path']
    gitlab_user = module.params['api_username']
    gitlab_password = module.params['api_password']
    gitlab_token = module.params['api_token']
    gitlab_oauth_token = module.params['api_oauth_token']
    gitlab_job_token = module.params['api_job_token']
    verify = ca_path if validate_certs and ca_path else validate_certs
    try:
        if LooseVersion(gitlab.__version__) < LooseVersion('1.13.0'):
            gitlab_instance = gitlab.Gitlab(url=gitlab_url, ssl_verify=verify, email=gitlab_user, password=gitlab_password, private_token=gitlab_token, api_version=4)
        else:
            if gitlab_user:
                data = {'grant_type': 'password', 'username': gitlab_user, 'password': gitlab_password}
                resp = requests.post(urljoin(gitlab_url, 'oauth/token'), data=data, verify=verify)
                resp_data = resp.json()
                gitlab_oauth_token = resp_data['access_token']
            gitlab_instance = gitlab.Gitlab(url=gitlab_url, ssl_verify=verify, private_token=gitlab_token, oauth_token=gitlab_oauth_token, job_token=gitlab_job_token, api_version=4)
        gitlab_instance.auth()
    except (gitlab.exceptions.GitlabAuthenticationError, gitlab.exceptions.GitlabGetError) as e:
        module.fail_json(msg='Failed to connect to GitLab server: %s' % to_native(e))
    except gitlab.exceptions.GitlabHttpError as e:
        module.fail_json(msg='Failed to connect to GitLab server: %s.             GitLab remove Session API now that private tokens are removed from user API endpoints since version 10.2.' % to_native(e))
    return gitlab_instance