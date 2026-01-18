from __future__ import annotations
import sys
from functools import partial
from typing import Any, Callable, Tuple, Type, cast
from attrs import fields, has, resolve_types
from cattrs import Converter
from cattrs.gen import (
from fontTools.misc.transform import Transform
def attrs_hook_factory(cls: Type[Any], gen_fn: Callable[..., Callable[..., Any]], structuring: bool) -> Callable[..., Any]:
    base = get_origin(cls)
    if base is None:
        base = cls
    attribs = fields(base)
    resolve_types(base)
    kwargs: dict[str, bool | AttributeOverride] = {'_cattrs_detailed_validation': conv.detailed_validation}
    if structuring:
        kwargs['_cattrs_forbid_extra_keys'] = conv.forbid_extra_keys
        kwargs['_cattrs_prefer_attrib_converters'] = conv._prefer_attrib_converters
    else:
        kwargs['_cattrs_omit_if_default'] = conv.omit_if_default
    for a in attribs:
        if a.type in conv.type_overrides:
            attrib_override = conv.type_overrides[a.type]
        else:
            attrib_override = override(omit_if_default=a.metadata.get('omit_if_default', a.default is None or None), rename=a.metadata.get('rename_attr', a.name[1:] if a.name[0] == '_' else None), omit=not a.init)
        kwargs[a.name] = attrib_override
    return gen_fn(cls, conv, **kwargs)