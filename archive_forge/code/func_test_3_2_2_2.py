from promise import Promise
from .utils import assert_exception
from threading import Event
def test_3_2_2_2():
    """
    Make sure callbacks are never called more than once.
    """
    c = Counter()
    p1 = Promise.resolve(5)
    p2 = p1.then(lambda v: c.tick())
    p2._wait()
    try:
        p1.do_resolve(5)
        assert False
    except AssertionError:
        pass
    assert 1 == c.value()