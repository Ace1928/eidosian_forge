import collections.abc
import dataclasses
import enum
import inspect
import os
import pathlib
from collections import deque
from typing import (
from typing_extensions import Annotated, Final, Literal, get_args, get_origin
from . import _resolver
from . import _strings
from ._typing import TypeForm
from .conf import _markers
def instantiator_from_type(typ: Union[TypeForm[Any], Callable], type_from_typevar: Dict[TypeVar, TypeForm[Any]], markers: FrozenSet[_markers.Marker]) -> Tuple[Instantiator, InstantiatorMetadata]:
    """Recursive helper for parsing type annotations.

    Returns two things:
    - An instantiator function, for instantiating the type from a string or list of
      strings. The latter applies when argparse's `nargs` parameter is set.
    - A metadata structure, which specifies parameters for argparse.
    """
    if typ is Any:
        raise UnsupportedTypeAnnotationError('`Any` is not a parsable type.')
    if typ is NoneType:

        def instantiator(strings: List[str]) -> None:
            assert strings == ['None']
        return (instantiator, InstantiatorMetadata(nargs=1, metavar='{None}', choices={'None'}, action=None))
    if typ is os.PathLike:
        typ = pathlib.Path
    container_out = _instantiator_from_container_type(cast(TypeForm[Any], typ), type_from_typevar, markers)
    if container_out is not None:
        return container_out
    metavar = getattr(typ, '__name__', '').upper()
    typ, maybe_newtype_name = _resolver.unwrap_newtype(typ)
    if maybe_newtype_name is not None:
        metavar = maybe_newtype_name.upper()
    if typ in _builtin_set:
        pass
    elif not callable(typ):
        raise UnsupportedTypeAnnotationError(f'Expected {typ} to be an `(arg: str) -> T` type converter, but is not callable.')
    elif not is_type_string_converter(typ):
        raise UnsupportedTypeAnnotationError(f'Expected {typ} to be an `(arg: str) -> T` type converter, but is not a valid type converter.')
    auto_choices: Optional[Tuple[str, ...]] = None
    if typ is bool:
        auto_choices = ('True', 'False')
    elif inspect.isclass(typ) and issubclass(typ, enum.Enum):
        auto_choices = tuple((x.name for x in typ))

    def instantiator_base_case(strings: List[str]) -> Any:
        """Given a type and and a string from the command-line, reconstruct an object. Not
        intended to deal with containers.

        This is intended to replace all calls to `type(string)`, which can cause unexpected
        behavior. As an example, note that the following argparse code will always print
        `True`, because `bool("True") == bool("False") == bool("0") == True`.
        ```
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--flag", type=bool)

        print(parser.parse_args().flag)
        ```
        """
        assert len(get_args(typ)) == 0, f'TypeForm {typ} cannot be instantiated.'
        string, = strings
        if typ is bool:
            return {'True': True, 'False': False}[string]
        elif isinstance(typ, type) and issubclass(typ, enum.Enum):
            return typ[string]
        elif typ is bytes:
            return bytes(string, encoding='ascii')
        else:
            return typ(string)
    return (instantiator_base_case, InstantiatorMetadata(nargs=1, metavar=metavar if auto_choices is None else '{' + ','.join(map(str, auto_choices)) + '}', choices=None if auto_choices is None else set(auto_choices), action=None))