from neutron_lib.api.definitions import bgpvpn
from neutron_lib.api import validators
from neutron_lib.tests.unit.api.definitions import base
def test_invalid_rtrd(self):
    for rtrd in self._data_for_invalid_rtdt():
        msg = validators.validate_list_of_regex_or_none(rtrd, bgpvpn.RTRD_REGEX)
        self.assertIsNotNone(msg)