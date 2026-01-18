from typing import TYPE_CHECKING, TypeVar, overload
def str_if_bytes(value: object) -> object:
    return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value