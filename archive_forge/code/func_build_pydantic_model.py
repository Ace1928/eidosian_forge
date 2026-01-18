import json
import typing
import datetime
import contextlib
from enum import Enum
from sqlalchemy import inspect
from lazyops.utils.serialization import object_serializer, Json
from sqlalchemy.ext.declarative import DeclarativeMeta
from pydantic import create_model, BaseModel, Field
from typing import Optional, Dict, Any, List, Union, Type, cast
def build_pydantic_model(obj: DeclarativeMeta) -> Type[BaseModel]:
    """
    Create a pydantic model from a sqlalchemy model
    """
    global _pydantic_models
    obj_class_name = f'{obj.__class__.__module__}.{obj.__class__.__name__}Model'
    if obj_class_name not in _pydantic_models:
        fields = get_model_fields(obj)
        _pydantic_models[obj_class_name] = create_model(f'{obj.__class__.__name__}Model', __config__=BasePydanticConfig, __module__=obj.__class__.__module__, **fields)
    return _pydantic_models[obj_class_name]