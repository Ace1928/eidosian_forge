from typing import MutableSequence, Optional, Sequence, TypeVar, Union, cast
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
def isfinished(self, item: Sequence[T]) -> bool:
    if not self.matchers:
        return True
    if self.mismatch_description:
        self.mismatch_description.append_text('no item matches: ').append_list('', ', ', '', self.matchers).append_text(' in ').append_list('[', ', ', ']', item)
    return False