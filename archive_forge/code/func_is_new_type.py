from dataclasses import InitVar
from typing import Type, Any, Optional, Union, Collection, TypeVar, Dict, Callable, Mapping, List, Tuple
def is_new_type(type_: Type) -> bool:
    return hasattr(type_, '__supertype__')