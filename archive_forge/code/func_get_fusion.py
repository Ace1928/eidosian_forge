from __future__ import absolute_import, division, print_function
from os import environ
from urllib.parse import urljoin
import platform
def get_fusion(module):
    """Return System Object or Fail"""
    _param_deprecation_warning(module, PARAM_APP_ID, PARAM_ISSUER_ID, DEP_VER)
    _param_deprecation_warning(module, PARAM_KEY_FILE, PARAM_PRIVATE_KEY_FILE, DEP_VER)
    _env_deprecation_warning(module, ENV_APP_ID, ENV_ISSUER_ID, DEP_VER)
    _env_deprecation_warning(module, ENV_HOST, ENV_API_HOST, DEP_VER)
    user_agent = '%(base)s %(class)s/%(version)s (%(platform)s)' % {'base': USER_AGENT_BASE, 'class': __name__, 'version': VERSION, 'platform': platform.platform()}
    issuer_id = module.params[PARAM_ISSUER_ID]
    access_token = module.params[PARAM_ACCESS_TOKEN]
    private_key_file = module.params[PARAM_PRIVATE_KEY_FILE]
    private_key_password = module.params[PARAM_PRIVATE_KEY_PASSWORD]
    if private_key_password is not None:
        module.fail_on_missing_params([PARAM_PRIVATE_KEY_FILE])
    config = fusion.Configuration()
    if ENV_API_HOST in environ or ENV_HOST in environ:
        host_url = environ.get(ENV_API_HOST, environ.get(ENV_HOST))
        config.host = urljoin(host_url, BASE_PATH)
    config.token_endpoint = environ.get(ENV_TOKEN_ENDPOINT, config.token_endpoint)
    if access_token is not None:
        config.access_token = access_token
    elif issuer_id is not None and private_key_file is not None:
        config.issuer_id = issuer_id
        config.private_key_file = private_key_file
        if private_key_password is not None:
            config.private_key_password = private_key_password
    elif ENV_ACCESS_TOKEN in environ:
        config.access_token = environ.get(ENV_ACCESS_TOKEN)
    elif (ENV_ISSUER_ID in environ or ENV_APP_ID in environ) and ENV_PRIVATE_KEY_FILE in environ:
        config.issuer_id = environ.get(ENV_ISSUER_ID, environ.get(ENV_APP_ID))
        config.private_key_file = environ.get(ENV_PRIVATE_KEY_FILE)
    else:
        module.fail_json(msg=f'You must set either {ENV_ISSUER_ID} and {ENV_PRIVATE_KEY_FILE} or {ENV_ACCESS_TOKEN} environment variables. Or module arguments either {PARAM_ISSUER_ID} and {PARAM_PRIVATE_KEY_FILE} or {PARAM_ACCESS_TOKEN}')
    try:
        client = fusion.ApiClient(config)
        client.set_default_header('User-Agent', user_agent)
        api_instance = fusion.DefaultApi(client)
        api_instance.get_version()
    except Exception as err:
        module.fail_json(msg='Fusion authentication failed: {0}'.format(err))
    return client