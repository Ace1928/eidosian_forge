import inspect
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.core.base.disable_methods import disable_methods
from pyomo.common.modeling import NOTSET
def test_bad_api(self):
    with self.assertRaisesRegex(DeveloperError, "Cannot disable method not_there on <class '.*\\.foo'>", normalize_whitespace=True):

        @disable_methods(('a', 'not_there'))
        class foo(_simple):
            pass