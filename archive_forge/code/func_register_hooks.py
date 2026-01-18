from __future__ import annotations
import sys
from functools import partial
from typing import Any, Callable, Tuple, Type, cast
from attrs import fields, has, resolve_types
from cattrs import Converter
from cattrs.gen import (
from fontTools.misc.transform import Transform
def register_hooks(conv: Converter, allow_bytes: bool=True) -> None:

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

    def custom_unstructure_hook_factory(cls: Type[Any]) -> Callable[[Any], Any]:
        return partial(cls._unstructure, converter=conv)

    def custom_structure_hook_factory(cls: Type[Any]) -> Callable[[Any, Any], Any]:
        return partial(cls._structure, converter=conv)

    def unstructure_transform(t: Transform) -> Tuple[float]:
        return cast(Tuple[float], tuple(t))
    conv.register_unstructure_hook_factory(is_ufoLib2_attrs_class, partial(attrs_hook_factory, gen_fn=make_dict_unstructure_fn, structuring=False))
    conv.register_unstructure_hook_factory(is_ufoLib2_class_with_custom_unstructure, custom_unstructure_hook_factory)
    conv.register_unstructure_hook(cast(Type[Transform], Transform), unstructure_transform)
    conv.register_structure_hook_factory(is_ufoLib2_attrs_class, partial(attrs_hook_factory, gen_fn=make_dict_structure_fn, structuring=True))
    conv.register_structure_hook_factory(is_ufoLib2_class_with_custom_structure, custom_structure_hook_factory)
    if not allow_bytes:
        from base64 import b64decode, b64encode

        def unstructure_bytes(v: bytes) -> str:
            return (b64encode(v) if v else b'').decode('utf8')

        def structure_bytes(v: str, _: Any) -> bytes:
            return b64decode(v)
        conv.register_unstructure_hook(bytes, unstructure_bytes)
        conv.register_structure_hook(bytes, structure_bytes)