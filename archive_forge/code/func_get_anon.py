from __future__ import annotations
import typing
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union
from ..util.typing import Literal
def get_anon(self, object_: Any) -> Tuple[str, bool]:
    idself = id(object_)
    if idself in self:
        s_val = self[idself]
        assert s_val is not True
        return (s_val, True)
    else:
        self[idself] = id_ = str(self._index)
        self._index += 1
        return (id_, False)