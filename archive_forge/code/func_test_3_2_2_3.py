from promise import Promise
from .utils import assert_exception
from threading import Event
def test_3_2_2_3():
    """
    Make sure fulfilled callback never called if promise is rejected
    """
    cf = Counter()
    cr = Counter()
    p1 = Promise.reject(Exception('Error'))
    p2 = p1.then(lambda v: cf.tick(), lambda r: cr.tick())
    p2._wait()
    assert 0 == cf.value()
    assert 1 == cr.value()