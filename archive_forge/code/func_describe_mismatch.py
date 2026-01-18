from typing import Any, List, Sequence, Tuple, TypeVar
from hamcrest import (
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.core.allof import AllOf
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
from typing_extensions import Protocol
from twisted.python.failure import Failure
def describe_mismatch(self, item: Sequence[T], description: Description) -> None:
    """
        Describe the mismatch.
        """
    for idx, elem in enumerate(item):
        if not self.elementMatcher.matches(elem):
            description.append_description_of(self)
            description.append_text(f'not sequence with element #{idx} {elem!r}')