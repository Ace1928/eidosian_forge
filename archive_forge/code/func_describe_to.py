from typing import Any, List, Sequence, Tuple, TypeVar
from hamcrest import (
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.core.allof import AllOf
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
from typing_extensions import Protocol
from twisted.python.failure import Failure
def describe_to(self, description: Description) -> None:
    """
        Describe this matcher for error messages.
        """
    description.append_text('a sequence containing only ')
    description.append_description_of(self.elementMatcher)
    description.append_text(', ')