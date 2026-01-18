from __future__ import annotations
import json
import contextlib
from abc import ABC
from urllib.parse import urljoin
from lazyops.libs import lazyload
from fastapi import Request
from fastapi.background import BackgroundTasks
from ..utils.lazy import get_az_settings, get_az_mtg_api, get_az_resource_schema, logger
from ..utils.helpers import get_hashed_key, create_code_challenge, parse_scopes, encode_params_to_url
from typing import Optional, List, Dict, Any, Union, Type
@property
def mtg_api(self) -> 'AZManagementAPI':
    """
        Returns the AZ Management API
        """
    if self._mtg_api is None:
        self._mtg_api = get_az_mtg_api()
    return self._mtg_api