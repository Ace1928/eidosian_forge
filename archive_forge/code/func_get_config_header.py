import base64
import json
import logging
from . import credentials
from . import errors
from .utils import config
def get_config_header(client, registry):
    log.debug('Looking for auth config')
    if not client._auth_configs or client._auth_configs.is_empty:
        log.debug('No auth config in memory - loading from filesystem')
        client._auth_configs = load_config(credstore_env=client.credstore_env)
    authcfg = resolve_authconfig(client._auth_configs, registry, credstore_env=client.credstore_env)
    if authcfg:
        log.debug('Found auth config')
        return encode_header(authcfg)
    log.debug('No auth config found')
    return None