from __future__ import annotations
import json
from typing import Any, Dict, Optional, Union, Type
from lazyops.utils.lazy import lazy_import
from .base import BaseSerializer, ObjectValue, SchemaType, BaseModel, logger
def decode_one(self, value: str, **kwargs) -> Union[SchemaType, Dict, Any]:
    """
        Decode the value with the JSON Library
        """
    try:
        value = self.jsonlib.loads(value, **kwargs)
        if not self.disable_object_serialization and isinstance(value, dict) and ('__class__' in value):
            obj_class_name = value.pop('__class__')
            if obj_class_name not in self.serialization_schemas:
                self.serialization_schemas[obj_class_name] = lazy_import(obj_class_name)
            obj_class = self.serialization_schemas[obj_class_name]
            value = obj_class.model_validate(value)
        elif self.serialization_obj is not None:
            value = self.serialization_obj.model_validate(value)
        return value
    except Exception as e:
        logger.info(f'Error Decoding Value: |r|({type(value)}) {e}|e| {str(value)[:1000]}', colored=True, prefix=self.jsonlib_name)
        if self.raise_errors:
            raise e
    return None