import typing
from lazyops.utils.imports import resolve_missing
import inspect
import pkg_resources
from pathlib import Path
from pydantic import BaseModel
from pydantic.fields import FieldInfo
def pyd_parse_obj(model: typing.Type[typing.Union[BaseModel, BaseSettings]], obj: typing.Any, **kwargs) -> typing.Union[BaseModel, BaseSettings]:
    """
    Parse an object into a pydantic model
    """
    return model.model_validate(obj, **kwargs) if PYD_VERSION == 2 else model.parse_obj(obj)