from neutron_lib.api.definitions import multiprovidernet
from neutron_lib.api.definitions import provider_net
from neutron_lib.exceptions import multiprovidernet as mp_exc
from neutron_lib.tests.unit.api.definitions import base
from neutron_lib.tests.unit.api.validators import test_multiprovidernet \
def test_check_duplicate_segments_with_dups_and_partial(self):

    def _seg_partial(seg):
        return seg[provider_net.PHYSICAL_NETWORK] == 'pn0'
    self.assertIsNone(multiprovidernet.check_duplicate_segments([test_mpnet._build_segment('nt0', 'pn0', 2), test_mpnet._build_segment('nt1', 'pn1', 2)], is_partial_func=_seg_partial))