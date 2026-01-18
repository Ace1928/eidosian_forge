from typing import Any, Callable, Literal, TypeVar
def is_jsonable(obj: Any) -> bool:
    """Check if an object is JSON serializable.

    This is a weak check, as it does not check for the actual JSON serialization, but only for the types of the object.
    It works correctly for basic use cases but do not guarantee an exhaustive check.

    Object is considered to be recursively json serializable if:
    - it is an instance of int, float, str, bool, or NoneType
    - it is a list or tuple and all its items are json serializable
    - it is a dict and all its keys are strings and all its values are json serializable
    """
    try:
        if isinstance(obj, _JSON_SERIALIZABLE_TYPES):
            return True
        if isinstance(obj, (list, tuple)):
            return all((is_jsonable(item) for item in obj))
        if isinstance(obj, dict):
            return all((isinstance(key, str) and is_jsonable(value) for key, value in obj.items()))
        if hasattr(obj, '__json__'):
            return True
        return False
    except RecursionError:
        return False