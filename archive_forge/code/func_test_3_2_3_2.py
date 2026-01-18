from promise import Promise
from .utils import assert_exception
from threading import Event
def test_3_2_3_2():
    """
    Make sure callbacks are never called more than once.
    """
    c = Counter()
    p1 = Promise.reject(Exception('Error'))
    p2 = p1.then(None, lambda v: c.tick())
    p2._wait()
    try:
        p1.do_reject(Exception('Error'))
        assert False
    except AssertionError:
        pass
    assert 1 == c.value()