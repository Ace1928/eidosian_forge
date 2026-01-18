from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def get_trust_tokens() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[TrustTokens]]:
    """
    Returns the number of stored Trust Tokens per issuer for the
    current browsing context.

    **EXPERIMENTAL**

    :returns: 
    """
    cmd_dict: T_JSON_DICT = {'method': 'Storage.getTrustTokens'}
    json = (yield cmd_dict)
    return [TrustTokens.from_json(i) for i in json['tokens']]