import functools
from neutron_lib._i18n import _
import neutron_lib.exceptions as ne
from neutron_lib.tests import _base as base
def test_invalid_content_type(self):
    self._check_nexc(ne.InvalidContentType, _('Invalid content type porn.'), content_type='porn')