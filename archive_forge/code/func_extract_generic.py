from dataclasses import InitVar
from typing import Type, Any, Optional, Union, Collection, TypeVar, Dict, Callable, Mapping, List, Tuple
def extract_generic(type_: Type, defaults: Tuple=()) -> tuple:
    try:
        if hasattr(type_, '_special') and type_._special:
            return defaults
        return type_.__args__ or defaults
    except AttributeError:
        return defaults