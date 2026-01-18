import os
import pathlib
from typing import Any, Type, Tuple, Dict, List, Union, Optional, Callable, TypeVar, Generic, TYPE_CHECKING
from lazyops.types.formatting import to_camel_case, to_snake_case, to_graphql_format
from lazyops.types.classprops import classproperty, lazyproperty
from lazyops.utils.serialization import Json
from pydantic import Field
from pydantic.networks import AnyUrl
from lazyops.imports._pydantic import BaseSettings as _BaseSettings
from lazyops.imports._pydantic import BaseModel as _BaseModel
from lazyops.imports._pydantic import (
@classmethod
def get_model_field_names(cls) -> List[str]:
    """
        Get the model fields
        """
    return get_pyd_field_names(cls)