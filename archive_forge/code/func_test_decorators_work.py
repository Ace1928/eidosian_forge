import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_decorators_work(self):

    @dummy_decorator
    def special_fun(x, y):
        pass
    result = reflection.get_callable_args(special_fun)
    self.assertEqual(['x', 'y'], result)