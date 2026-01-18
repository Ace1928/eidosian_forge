from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def get_oci_config(module, service_client_class=None):
    """Return the OCI configuration to use for all OCI API calls. The effective OCI configuration is derived by merging
    any overrides specified for configuration attributes through Ansible module options or environment variables. The
    order of precedence for deriving the effective configuration dict is:
    1. If a config file is provided, use that to setup the initial config dict.
    2. If a config profile is specified, use that config profile to setup the config dict.
    3. For each authentication attribute, check if an override is provided either through
        a. Ansible Module option
        b. Environment variable
        and override the value in the config dict in that order."""
    config = {}
    config_file = module.params.get('config_file_location')
    _debug('Config file through module options - {0} '.format(config_file))
    if not config_file:
        if 'OCI_CONFIG_FILE' in os.environ:
            config_file = os.environ['OCI_CONFIG_FILE']
            _debug('Config file through OCI_CONFIG_FILE environment variable - {0}'.format(config_file))
        else:
            config_file = '~/.oci/config'
            _debug('Config file (fallback) - {0} '.format(config_file))
    config_profile = module.params.get('config_profile_name')
    if not config_profile:
        if 'OCI_CONFIG_PROFILE' in os.environ:
            config_profile = os.environ['OCI_CONFIG_PROFILE']
        else:
            config_profile = 'DEFAULT'
    try:
        config = oci.config.from_file(file_location=config_file, profile_name=config_profile)
    except (ConfigFileNotFound, InvalidConfig, InvalidPrivateKey, MissingPrivateKeyPassphrase) as ex:
        if not _is_instance_principal_auth(module):
            module.fail_json(msg=str(ex))
        else:
            _debug('Ignore {0} as the auth_type is set to instance_principal'.format(str(ex)))
    config['additional_user_agent'] = 'Oracle-Ansible/{0}'.format(__version__)
    _merge_auth_option(config, module, module_option_name='api_user', env_var_name='OCI_USER_ID', config_attr_name='user')
    _merge_auth_option(config, module, module_option_name='api_user_fingerprint', env_var_name='OCI_USER_FINGERPRINT', config_attr_name='fingerprint')
    _merge_auth_option(config, module, module_option_name='api_user_key_file', env_var_name='OCI_USER_KEY_FILE', config_attr_name='key_file')
    _merge_auth_option(config, module, module_option_name='api_user_key_pass_phrase', env_var_name='OCI_USER_KEY_PASS_PHRASE', config_attr_name='pass_phrase')
    _merge_auth_option(config, module, module_option_name='tenancy', env_var_name='OCI_TENANCY', config_attr_name='tenancy')
    _merge_auth_option(config, module, module_option_name='region', env_var_name='OCI_REGION', config_attr_name='region')
    do_not_redirect = module.params.get('do_not_redirect_to_home_region', False) or os.environ.get('OCI_IDENTITY_DO_NOT_REDIRECT_TO_HOME_REGION')
    if service_client_class == IdentityClient and (not do_not_redirect):
        _debug('Region passed for module invocation - {0} '.format(config['region']))
        identity_client = IdentityClient(config)
        region_subscriptions = identity_client.list_region_subscriptions(config['tenancy']).data
        [config['region']] = [rs.region_name for rs in region_subscriptions if rs.is_home_region is True]
        _debug('Setting region in the config to home region - {0} '.format(config['region']))
    return config