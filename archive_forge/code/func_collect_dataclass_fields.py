from __future__ import annotations as _annotations
import dataclasses
import sys
import warnings
from copy import copy
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from pydantic_core import PydanticUndefined
from pydantic.errors import PydanticUserError
from . import _typing_extra
from ._config import ConfigWrapper
from ._repr import Representation
from ._typing_extra import get_cls_type_hints_lenient, get_type_hints, is_classvar, is_finalvar
def collect_dataclass_fields(cls: type[StandardDataclass], types_namespace: dict[str, Any] | None, *, typevars_map: dict[Any, Any] | None=None) -> dict[str, FieldInfo]:
    """Collect the fields of a dataclass.

    Args:
        cls: dataclass.
        types_namespace: Optional extra namespace to look for types in.
        typevars_map: A dictionary mapping type variables to their concrete types.

    Returns:
        The dataclass fields.
    """
    from ..fields import FieldInfo
    fields: dict[str, FieldInfo] = {}
    dataclass_fields: dict[str, dataclasses.Field] = cls.__dataclass_fields__
    cls_localns = dict(vars(cls))
    source_module = sys.modules.get(cls.__module__)
    if source_module is not None:
        types_namespace = {**source_module.__dict__, **(types_namespace or {})}
    for ann_name, dataclass_field in dataclass_fields.items():
        ann_type = _typing_extra.eval_type_lenient(dataclass_field.type, types_namespace, cls_localns)
        if is_classvar(ann_type):
            continue
        if not dataclass_field.init and dataclass_field.default == dataclasses.MISSING and (dataclass_field.default_factory == dataclasses.MISSING):
            continue
        if isinstance(dataclass_field.default, FieldInfo):
            if dataclass_field.default.init_var:
                if dataclass_field.default.init is False:
                    raise PydanticUserError(f'Dataclass field {ann_name} has init=False and init_var=True, but these are mutually exclusive.', code='clashing-init-and-init-var')
                continue
            field_info = FieldInfo.from_annotated_attribute(ann_type, dataclass_field.default)
        else:
            field_info = FieldInfo.from_annotated_attribute(ann_type, dataclass_field)
        fields[ann_name] = field_info
        if field_info.default is not PydanticUndefined and isinstance(getattr(cls, ann_name, field_info), FieldInfo):
            setattr(cls, ann_name, field_info.default)
    if typevars_map:
        for field in fields.values():
            field.apply_typevars_map(typevars_map, types_namespace)
    return fields