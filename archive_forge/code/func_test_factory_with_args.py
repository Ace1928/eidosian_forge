from osprofiler.drivers import base
from osprofiler.tests import test
def test_factory_with_args(self):

    class B(base.Driver):

        def __init__(self, c_str, a, b=10):
            self.a = a
            self.b = b

        @classmethod
        def get_name(cls):
            return 'b'

        def notify(self, c):
            return self.a + self.b + c
    self.assertEqual(22, base.get_driver('b://', 5, b=7).notify(10))