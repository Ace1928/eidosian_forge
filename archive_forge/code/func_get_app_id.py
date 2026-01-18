from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def get_app_id() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[typing.Optional[str], typing.Optional[str]]]:
    """
    Returns the unique (PWA) app id.
    Only returns values if the feature flag 'WebAppEnableManifestId' is enabled

    **EXPERIMENTAL**

    :returns: A tuple with the following items:

        0. **appId** - *(Optional)* App id, either from manifest's id attribute or computed from start_url
        1. **recommendedId** - *(Optional)* Recommendation for manifest's id attribute to match current id computed from start_url
    """
    cmd_dict: T_JSON_DICT = {'method': 'Page.getAppId'}
    json = (yield cmd_dict)
    return (str(json['appId']) if 'appId' in json else None, str(json['recommendedId']) if 'recommendedId' in json else None)