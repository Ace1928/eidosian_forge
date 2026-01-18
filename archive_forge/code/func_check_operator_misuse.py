from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from langchain_community.utilities.redis import TokenEscaper
def check_operator_misuse(func: Callable) -> Callable:
    """Decorator to check for misuse of equality operators."""

    @wraps(func)
    def wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
        other = kwargs.get('other') if 'other' in kwargs else None
        if not other:
            for arg in args:
                if isinstance(arg, type(instance)):
                    other = arg
                    break
        if isinstance(other, type(instance)):
            raise ValueError('Equality operators are overridden for FilterExpression creation. Use .equals() for equality checks')
        return func(instance, *args, **kwargs)
    return wrapper