from dataclasses import InitVar
from typing import Type, Any, Optional, Union, Collection, TypeVar, Dict, Callable, Mapping, List, Tuple
def extract_init_var(type_: Type) -> Union[Type, Any]:
    try:
        return type_.type
    except AttributeError:
        return Any