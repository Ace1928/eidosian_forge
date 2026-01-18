from dataclasses import InitVar
from typing import Type, Any, Optional, Union, Collection, TypeVar, Dict, Callable, Mapping, List, Tuple
def is_init_var(type_: Type) -> bool:
    return isinstance(type_, InitVar) or type_ is InitVar