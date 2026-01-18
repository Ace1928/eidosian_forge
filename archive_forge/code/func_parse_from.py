from lazyops.types.models import BaseModel
from lazyops.types.static import VALID_REQUEST_KWARGS
from lazyops.types.classprops import lazyproperty
from lazyops.imports._aiohttpx import aiohttpx
from lazyops.imports._pydantic import get_pyd_fields, pyd_parse_obj, get_pyd_dict
from typing import Optional, Dict, Any, Tuple, Type, Union, TYPE_CHECKING
@classmethod
def parse_from(cls, input_obj: ResourceType, response: 'aiohttpx.Response', **kwargs) -> 'ResponseResourceType':
    """
        Parses the response and returns the response object
        """
    resp = cls(_input_obj=input_obj, _response=response, **response.json())
    resp.parse_data(**kwargs)
    return resp