from __future__ import annotations
import socket
import contextlib
from abc import ABC
from urllib.parse import urljoin
from lazyops.libs import lazyload
from lazyops.utils.helpers import timed_cache
from lazyops.libs.abcs.configs.types import AppEnv
from ..utils.lazy import get_az_settings, logger, get_az_flow, get_az_resource
from typing import Optional, List, Dict, Any, Union
def check_authorization(self):
    """
        Checks authorization
        """
    if self._authorized:
        return
    try:
        response = self.session.get(self.get_url('/authorize'), timeout=2.5, auth=self.auth)
        response.raise_for_status()
    except niquests.HTTPError as e:
        logger.error(f'Error Authorizing Client: {e}')
        return
    data: Dict[str, Any] = response.json()
    if (api_key := data.get('api_key', data.get('api-key'))):
        self._api_key = api_key
        self.data['api_key'] = api_key
        self.auth.x_api_key = api_key
    if (api_env := data.get('environment')):
        self.data['environment'] = api_env
    if (identity := data.get('identity')):
        self.data['identity'] = identity
    self._authorized = True
    self.token_flow.save_data(self.data)