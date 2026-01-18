from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
def get_heroku_client(self):
    client = heroku3.from_key(self.api_key)
    if not client.is_authenticated:
        self.module.fail_json(msg='Heroku authentication failure, please check your API Key')
    return client