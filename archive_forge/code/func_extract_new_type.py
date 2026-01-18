from dataclasses import InitVar
from typing import Type, Any, Optional, Union, Collection, TypeVar, Dict, Callable, Mapping, List, Tuple
def extract_new_type(type_: Type) -> Type:
    return type_.__supertype__