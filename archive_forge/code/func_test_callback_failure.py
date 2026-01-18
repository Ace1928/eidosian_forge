import functools
import neutron_lib.callbacks.exceptions as ex
from neutron_lib.tests.unit.exceptions import test_exceptions
def test_callback_failure(self):
    self._check_exception(ex.CallbackFailure, 'one', errors='one')