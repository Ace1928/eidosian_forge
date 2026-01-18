import pytest
from requests.structures import CaseInsensitiveDict, LookupDict
def test_lower_items(self):
    assert list(self.case_insensitive_dict.lower_items()) == [('accept', 'application/json')]