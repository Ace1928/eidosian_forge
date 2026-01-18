from typing import Any, Type, TypeVar
from triad.utils.assertion import assert_or_throw
def import_fsql_dependency(package_name: str) -> Any:
    return import_or_throw(package_name, 'Please try to install the package by `pip install fugue[sql]`.')