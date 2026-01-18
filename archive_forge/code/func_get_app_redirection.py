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
def get_app_redirection(self, redirect: str) -> Union[str, 'URLPath']:
    """
        Gets the app redirection
        """
    if redirect.startswith('http'):
        return redirect
    if 'docs=' in redirect:
        base_url = str(self.app.url_path_for('docs').make_absolute_url(self.app_ingress)) + '#/operations'
        redirect = redirect.replace('docs=', '')
        if redirect in self.docs_schema_index:
            return f'{base_url}/{self.docs_schema_index[redirect]}'
    return self.app.url_path_for(redirect).make_absolute_url(self.app_ingress)