from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, TypeVar, cast
from pymongo.server_type import SERVER_TYPE
def arbiter_server_selector(selection: Selection) -> Selection:
    return selection.with_server_descriptions([s for s in selection.server_descriptions if s.server_type == SERVER_TYPE.RSArbiter])