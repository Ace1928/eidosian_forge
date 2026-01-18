from lazyops.types.models import BaseModel
from lazyops.types.static import VALID_REQUEST_KWARGS
from lazyops.types.classprops import lazyproperty
from lazyops.imports._aiohttpx import aiohttpx
from lazyops.imports._pydantic import get_pyd_fields, pyd_parse_obj, get_pyd_dict
from typing import Optional, Dict, Any, Tuple, Type, Union, TYPE_CHECKING
@classmethod
def parse_resource(cls, **kwargs) -> Tuple[Type['BaseResource'], Dict]:
    """
        Extracts the resource from the kwargs and returns the resource 
        and the remaining kwargs
        """
    resource_fields = [field.name for field in get_pyd_fields(cls)]
    resource_kwargs = {k: v for k, v in kwargs.items() if k in resource_fields}
    return_kwargs = {k: v for k, v in kwargs.items() if k not in resource_fields and k in VALID_REQUEST_KWARGS}
    resource_obj = pyd_parse_obj(cls, resource_kwargs)
    return (resource_obj, return_kwargs)