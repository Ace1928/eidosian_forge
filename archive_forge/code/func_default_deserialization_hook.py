from __future__ import annotations
from typing import Any, Dict, Optional, Union, Type
from lazyops.utils.lazy import lazy_import
from .base import BinaryBaseSerializer, BaseModel, SchemaType, ObjectValue, logger
from ._json import default_json
def default_deserialization_hook(self, code: int, data: Union[str, bytes]) -> ObjectValue:
    """
        Default Deserialization Hook
        """
    if code != 2:
        return data
    if isinstance(data, bytes):
        data = data.decode(self.encoding)
    try:
        data = self.jsonlib.loads(data)
    except Exception as e:
        logger.info(f'Error Decoding Value: |r|({type(data)}) {e}|e| {str(data)[:500]}', colored=True, prefix='msgpack')
        if self.raise_errors:
            raise e
        return data
    if not self.disable_object_serialization:
        _class = data.pop('__class__', None)
        if _class is not None:
            if _class not in self.serialization_schemas:
                self.serialization_schemas[_class] = lazy_import(_class)
            _class = self.serialization_schemas[_class]
            return _class.model_validate(data, **self.serialization_obj_kwargs)
    elif self.serialization_obj is not None:
        return self.serialization_obj.model_validate(data, **self.serialization_obj_kwargs)
    return data