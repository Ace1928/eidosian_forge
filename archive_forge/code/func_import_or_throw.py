from typing import Any, Type, TypeVar
from triad.utils.assertion import assert_or_throw
def import_or_throw(package_name: str, message: str) -> Any:
    try:
        return __import__(package_name)
    except Exception as e:
        raise ImportError(str(e) + '. ' + message)