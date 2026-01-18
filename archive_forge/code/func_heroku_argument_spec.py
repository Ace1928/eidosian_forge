from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
@staticmethod
def heroku_argument_spec():
    return dict(api_key=dict(fallback=(env_fallback, ['HEROKU_API_KEY', 'TF_VAR_HEROKU_API_KEY']), type='str', no_log=True))