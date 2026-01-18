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
def get_app_manifest() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[str, typing.List[AppManifestError], typing.Optional[str], typing.Optional[AppManifestParsedProperties]]]:
    """


    :returns: A tuple with the following items:

        0. **url** - Manifest location.
        1. **errors** - 
        2. **data** - *(Optional)* Manifest content.
        3. **parsed** - *(Optional)* Parsed manifest properties
    """
    cmd_dict: T_JSON_DICT = {'method': 'Page.getAppManifest'}
    json = (yield cmd_dict)
    return (str(json['url']), [AppManifestError.from_json(i) for i in json['errors']], str(json['data']) if 'data' in json else None, AppManifestParsedProperties.from_json(json['parsed']) if 'parsed' in json else None)