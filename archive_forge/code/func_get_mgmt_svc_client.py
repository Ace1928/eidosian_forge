from __future__ import absolute_import, division, print_function
import os
import re
import types
import copy
import inspect
import traceback
import json
from os.path import expanduser
from ansible.module_utils.basic import \
from ansible.module_utils.six.moves import configparser
import ansible.module_utils.six.moves.urllib.parse as urlparse
from base64 import b64encode, b64decode
from hashlib import sha256
from hmac import HMAC
from time import time
def get_mgmt_svc_client(self, client_type, base_url=None, api_version=None, suppress_subscription_id=False, is_track2=False):
    self.log('Getting management service client {0}'.format(client_type.__name__))
    self.check_client_version(client_type)
    client_argspec = inspect.signature(client_type.__init__)
    if not base_url:
        base_url = self.azure_auth._cloud_environment.endpoints.resource_manager
    if not base_url.endswith('/'):
        base_url += '/'
    mgmt_subscription_id = self.azure_auth.subscription_id
    if self.module.params.get('subscription_id'):
        mgmt_subscription_id = self.module.params.get('subscription_id')
    if suppress_subscription_id:
        if is_track2:
            client_kwargs = dict(credential=self.azure_auth.azure_credential_track2, base_url=base_url, credential_scopes=[base_url + '.default'])
        else:
            client_kwargs = dict(credentials=self.azure_auth.azure_credentials, base_url=base_url)
    elif is_track2:
        client_kwargs = dict(credential=self.azure_auth.azure_credential_track2, subscription_id=mgmt_subscription_id, base_url=base_url, credential_scopes=[base_url + '.default'])
    else:
        client_kwargs = dict(credentials=self.azure_auth.azure_credentials, subscription_id=mgmt_subscription_id, base_url=base_url)
    api_profile_dict = {}
    if self.api_profile:
        api_profile_dict = self.get_api_profile(client_type.__name__, self.api_profile)
    if api_profile_dict and 'profile' in client_argspec.parameters:
        client_kwargs['profile'] = api_profile_dict
    if 'api_version' in client_argspec.parameters:
        profile_default_version = api_profile_dict.get('default_api_version', None)
        if api_version or profile_default_version:
            client_kwargs['api_version'] = api_version or profile_default_version
            if 'profile' in client_kwargs:
                client_kwargs.pop('profile')
    client = client_type(**client_kwargs)
    try:
        getattr(client, 'models')
    except AttributeError:

        def _ansible_get_models(self, *arg, **kwarg):
            return self._ansible_models
        setattr(client, '_ansible_models', importlib.import_module(client_type.__module__).models)
        client.models = types.MethodType(_ansible_get_models, client)
    if not is_track2:
        client.config = self.add_user_agent(client.config)
        if self.azure_auth._cert_validation_mode == 'ignore':
            client.config.session_configuration_callback = self._validation_ignore_callback
    elif self.azure_auth._cert_validation_mode == 'ignore':
        client._config.session_configuration_callback = self._validation_ignore_callback
    return client