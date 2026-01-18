import keyword
import warnings
import weakref
from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from itertools import islice, zip_longest
from types import BuiltinFunctionType, CodeType, FunctionType, GeneratorType, LambdaType, ModuleType
from typing import (
from typing_extensions import Annotated
from .errors import ConfigError
from .typing import (
from .version import version_info
def get_discriminator_alias_and_values(tp: Any, discriminator_key: str) -> Tuple[str, Tuple[str, ...]]:
    """
    Get alias and all valid values in the `Literal` type of the discriminator field
    `tp` can be a `BaseModel` class or directly an `Annotated` `Union` of many.
    """
    is_root_model = getattr(tp, '__custom_root_type__', False)
    if get_origin(tp) is Annotated:
        tp = get_args(tp)[0]
    if hasattr(tp, '__pydantic_model__'):
        tp = tp.__pydantic_model__
    if is_union(get_origin(tp)):
        alias, all_values = _get_union_alias_and_all_values(tp, discriminator_key)
        return (alias, tuple((v for values in all_values for v in values)))
    elif is_root_model:
        union_type = tp.__fields__[ROOT_KEY].type_
        alias, all_values = _get_union_alias_and_all_values(union_type, discriminator_key)
        if len(set(all_values)) > 1:
            raise ConfigError(f'Field {discriminator_key!r} is not the same for all submodels of {display_as_type(tp)!r}')
        return (alias, all_values[0])
    else:
        try:
            t_discriminator_type = tp.__fields__[discriminator_key].type_
        except AttributeError as e:
            raise TypeError(f'Type {tp.__name__!r} is not a valid `BaseModel` or `dataclass`') from e
        except KeyError as e:
            raise ConfigError(f'Model {tp.__name__!r} needs a discriminator field for key {discriminator_key!r}') from e
        if not is_literal_type(t_discriminator_type):
            raise ConfigError(f'Field {discriminator_key!r} of model {tp.__name__!r} needs to be a `Literal`')
        return (tp.__fields__[discriminator_key].alias, all_literal_values(t_discriminator_type))