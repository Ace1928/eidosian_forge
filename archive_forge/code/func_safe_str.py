from typing import TYPE_CHECKING, TypeVar, overload
def safe_str(value: object) -> str:
    return str(str_if_bytes(value))