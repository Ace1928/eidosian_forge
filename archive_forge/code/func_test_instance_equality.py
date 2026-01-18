import pytest
from requests.structures import CaseInsensitiveDict, LookupDict
@pytest.mark.parametrize('other, result', (({'AccePT': 'application/json'}, True), ({}, False), (None, False)))
def test_instance_equality(self, other, result):
    assert (self.case_insensitive_dict == other) is result