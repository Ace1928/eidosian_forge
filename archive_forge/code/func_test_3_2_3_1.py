from promise import Promise
from .utils import assert_exception
from threading import Event
def test_3_2_3_1():
    """
    The second argument to 'then' must be called when a promise is
    rejected.
    """
    c = Counter()

    def check(r, c):
        assert_exception(r, Exception, 'Error')
        c.tick()
    p1 = Promise.reject(Exception('Error'))
    p2 = p1.then(None, lambda r: check(r, c))
    p2._wait()
    assert 1 == c.value()