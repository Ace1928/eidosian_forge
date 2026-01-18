from __future__ import annotations
import argparse
import json
from typing import Any
import requests
import extruct
from extruct import SYNTAXES
def metadata_from_url(url: str, syntaxes: list[str]=SYNTAXES, uniform: bool=False, schema_context: str='http://schema.org', errors: str='strict') -> dict[str, Any]:
    resp = requests.get(url, timeout=30)
    result: dict[str, Any] = {'url': url, 'status': '{} {}'.format(resp.status_code, resp.reason)}
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        return result
    result.update(extruct.extract(resp.content, base_url=url, syntaxes=syntaxes, uniform=uniform, schema_context=schema_context, errors=errors))
    return result