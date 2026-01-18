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
def generate_model_signature(init: Callable[..., None], fields: Dict[str, 'ModelField'], config: Type['BaseConfig']) -> 'Signature':
    """
    Generate signature for model based on its fields
    """
    from inspect import Parameter, Signature, signature
    from .config import Extra
    present_params = signature(init).parameters.values()
    merged_params: Dict[str, Parameter] = {}
    var_kw = None
    use_var_kw = False
    for param in islice(present_params, 1, None):
        if param.kind is param.VAR_KEYWORD:
            var_kw = param
            continue
        merged_params[param.name] = param
    if var_kw:
        allow_names = config.allow_population_by_field_name
        for field_name, field in fields.items():
            param_name = field.alias
            if field_name in merged_params or param_name in merged_params:
                continue
            elif not is_valid_identifier(param_name):
                if allow_names and is_valid_identifier(field_name):
                    param_name = field_name
                else:
                    use_var_kw = True
                    continue
            kwargs = {'default': field.default} if not field.required else {}
            merged_params[param_name] = Parameter(param_name, Parameter.KEYWORD_ONLY, annotation=field.annotation, **kwargs)
    if config.extra is Extra.allow:
        use_var_kw = True
    if var_kw and use_var_kw:
        default_model_signature = [('__pydantic_self__', Parameter.POSITIONAL_OR_KEYWORD), ('data', Parameter.VAR_KEYWORD)]
        if [(p.name, p.kind) for p in present_params] == default_model_signature:
            var_kw_name = 'extra_data'
        else:
            var_kw_name = var_kw.name
        while var_kw_name in fields:
            var_kw_name += '_'
        merged_params[var_kw_name] = var_kw.replace(name=var_kw_name)
    return Signature(parameters=list(merged_params.values()), return_annotation=None)