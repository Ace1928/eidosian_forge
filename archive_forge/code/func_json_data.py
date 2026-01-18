from lazyops.types.models import BaseModel
from lazyops.types.static import VALID_REQUEST_KWARGS
from lazyops.types.classprops import lazyproperty
from lazyops.imports._aiohttpx import aiohttpx
from lazyops.imports._pydantic import get_pyd_fields, pyd_parse_obj, get_pyd_dict
from typing import Optional, Dict, Any, Tuple, Type, Union, TYPE_CHECKING
@lazyproperty
def json_data(self) -> Dict[str, Any]:
    return self._response.json()