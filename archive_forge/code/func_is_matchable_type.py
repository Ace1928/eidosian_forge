from typing import Type, TypeVar, Union
from hamcrest.core.base_matcher import Matcher
from hamcrest.core.core.isequal import equal_to
def is_matchable_type(expected_type: Type) -> bool:
    if isinstance(expected_type, type):
        return True
    return False