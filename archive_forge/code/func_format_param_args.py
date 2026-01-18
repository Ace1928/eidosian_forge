import enum
import typing
from functools import lru_cache
from typing import Any, Dict, FrozenSet, Iterable, List, Tuple, Union
def format_param_args(param: ParamName, details: Any) -> Union[ParamArg, List[ParamVal]]:
    """Ensures each user-inputted parameter is a properly typed list.
    Also provides custom support for certain parameters."""
    if not isinstance(details, list):
        details = [details]
    for detail in details:
        if ParamArg.is_arg(detail):
            return ParamArg(detail)
    if param == 'layout':
        if all((isinstance(dim, int) for dim in details)):
            return ['x'.join(map(str, details))]
        for i, detail in enumerate(details):
            if isinstance(detail, Iterable) and all((isinstance(dim, int) for dim in detail)):
                details[i] = 'x'.join(map(str, detail))
            elif not isinstance(detail, str):
                raise TypeError(f"Invalid layout value of '{detail}'. Must be a string or a tuple of ints.")
    elif param == 'bondlength':
        for i, detail in enumerate(details):
            if isinstance(detail, float):
                details[i] = str(detail)
            elif isinstance(detail, int):
                details[i] = f'{detail:.1f}'
            elif not isinstance(detail, str):
                raise TypeError(f"Invalid bondlength '{detail}'. Must be a string, int or float.")
    for detail in details:
        if not isinstance(detail, str):
            raise TypeError(f"Invalid type '{type(detail).__name__}' for parameter '{param}'")
    return details