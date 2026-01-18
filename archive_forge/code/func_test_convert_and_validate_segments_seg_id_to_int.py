from webob import exc
from neutron_lib.api.definitions import provider_net
from neutron_lib.api.validators import multiprovidernet as mp_validator
from neutron_lib import constants
from neutron_lib.tests import _base as base
def test_convert_and_validate_segments_seg_id_to_int(self):
    segs = [_build_segment(seg_id='9')]
    mp_validator.convert_and_validate_segments(segs)
    self.assertEqual(_build_segment(seg_id=9), segs[0])