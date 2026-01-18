from typing import Hashable, Mapping, MutableMapping, TypeVar, Union
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.helpers.hasmethod import hasmethod
from hamcrest.core.helpers.wrap_matcher import wrap_matcher
from hamcrest.core.matcher import Matcher
def describe_match(self, item: Mapping[K, V], match_description: Description) -> None:
    key_matches = self._matching_keys(item)
    if len(key_matches) == 1:
        key, value = key_matches.popitem()
        match_description.append_text('value for ').append_description_of(key).append_text(' ')
        self.value_matcher.describe_mismatch(value, match_description)
    else:
        super().describe_match(item, match_description)