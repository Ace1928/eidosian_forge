import gc
from io import StringIO
import weakref
from pyomo.common.unittest import TestCase
from pyomo.common.log import LoggingIntercept
from pyomo.common.plugin_base import (
def test_inherit_interface(self):

    class IFoo(Interface):

        def fcn(self):
            return 'base'

        def baseFcn(self):
            return 'baseFcn'

    class myFoo(Plugin):
        implements(IFoo)
    with self.assertRaises(AttributeError):
        myFoo().fcn()

    class myFoo(Plugin):
        implements(IFoo, inherit=True)

        def fcn(self):
            return 'derived'
    self.assertEqual(myFoo().fcn(), 'derived')
    self.assertEqual(myFoo().baseFcn(), 'baseFcn')

    class IMock(Interface):

        def mock(self):
            return 'mock'

    class myCombined(myFoo):
        implements(IMock, inherit=True)
    a = myCombined()
    self.assertEqual(a.fcn(), 'derived')
    self.assertEqual(a.baseFcn(), 'baseFcn')
    self.assertEqual(a.mock(), 'mock')