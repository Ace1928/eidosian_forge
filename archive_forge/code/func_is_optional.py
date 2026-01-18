from dataclasses import InitVar
from typing import Type, Any, Optional, Union, Collection, TypeVar, Dict, Callable, Mapping, List, Tuple
def is_optional(type_: Type) -> bool:
    return is_union(type_) and type(None) in extract_generic(type_)