from __future__ import absolute_import, division, print_function
import base64
import json
import os
import traceback
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api import auth
from ansible_collections.community.docker.plugins.module_utils._api.auth import decode_auth
from ansible_collections.community.docker.plugins.module_utils._api.credentials.errors import CredentialsNotFound
from ansible_collections.community.docker.plugins.module_utils._api.credentials.store import Store
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
def get_credential_store_instance(self, registry, dockercfg_path):
    """
        Return an instance of docker.credentials.Store used by the given registry.

        :return: A Store or None
        :rtype: Union[docker.credentials.Store, NoneType]
        """
    credstore_env = self.client.credstore_env
    config = auth.load_config(config_path=dockercfg_path)
    store_name = auth.get_credential_store(config, registry)
    if store_name:
        self.log('Found credential store %s' % store_name)
        return Store(store_name, environment=credstore_env)
    return DockerFileStore(dockercfg_path)