from __future__ import annotations
from operator import attrgetter
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Optional
from typing import Type
from typing import Union
from . import url as _url
from .. import util
def schema_for_object(self, obj: HasSchemaAttr) -> Optional[str]:
    return obj.schema