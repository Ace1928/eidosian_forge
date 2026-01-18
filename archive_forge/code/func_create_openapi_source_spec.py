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
def create_openapi_source_spec(self, spec: Dict[str, Any], spec_map: Optional[Dict[str, Any]]=None):
    """
        Creates the Source Spec

        - Handles some custom logic for the OpenAPI Spec
            Namely:
            - AppResponse-Input -> AppResponse
            - AppResponse-Output -> AppResponse
        """
    if not spec_map:
        return
    _spec = json.dumps(spec)
    for key, value in spec_map.items():
        _spec = _spec.replace(key, value)
    self.source_openapi_schema = json.loads(_spec)