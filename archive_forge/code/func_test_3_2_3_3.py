from promise import Promise
from .utils import assert_exception
from threading import Event
def test_3_2_3_3():
    """
    Make sure rejected callback never called if promise is fulfilled
    """
    cf = Counter()
    cr = Counter()
    p1 = Promise.resolve(5)
    p2 = p1.then(lambda v: cf.tick(), lambda r: cr.tick())
    p2._wait()
    assert 0 == cr.value()
    assert 1 == cf.value()