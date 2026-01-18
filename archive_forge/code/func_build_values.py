from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, TypeVar, Union, overload
from . import validator
from .config import Extra
from .errors import ConfigError
from .main import BaseModel, create_model
from .typing import get_all_type_hints
from .utils import to_camel
def build_values(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    if args:
        arg_iter = enumerate(args)
        while True:
            try:
                i, a = next(arg_iter)
            except StopIteration:
                break
            arg_name = self.arg_mapping.get(i)
            if arg_name is not None:
                values[arg_name] = a
            else:
                values[self.v_args_name] = [a] + [a for _, a in arg_iter]
                break
    var_kwargs: Dict[str, Any] = {}
    wrong_positional_args = []
    duplicate_kwargs = []
    fields_alias = [field.alias for name, field in self.model.__fields__.items() if name not in (self.v_args_name, self.v_kwargs_name)]
    non_var_fields = set(self.model.__fields__) - {self.v_args_name, self.v_kwargs_name}
    for k, v in kwargs.items():
        if k in non_var_fields or k in fields_alias:
            if k in self.positional_only_args:
                wrong_positional_args.append(k)
            if k in values:
                duplicate_kwargs.append(k)
            values[k] = v
        else:
            var_kwargs[k] = v
    if var_kwargs:
        values[self.v_kwargs_name] = var_kwargs
    if wrong_positional_args:
        values[V_POSITIONAL_ONLY_NAME] = wrong_positional_args
    if duplicate_kwargs:
        values[V_DUPLICATE_KWARGS] = duplicate_kwargs
    return values