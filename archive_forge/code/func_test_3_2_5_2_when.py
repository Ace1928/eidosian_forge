from promise import Promise
from .utils import assert_exception
from threading import Event
def test_3_2_5_2_when():
    """
    Then can be called multiple times on the same promise
    and callbacks must be called in the order of the
    then calls.
    """

    def add(l, v):
        l.append(v)
    p1 = Promise.reject(Exception('Error'))
    order = []
    p2 = p1.then(None, lambda v: add(order, 'p2'))
    p3 = p1.then(None, lambda v: add(order, 'p3'))
    p2._wait()
    p3._wait()
    assert 2 == len(order)
    assert 'p2' == order[0]
    assert 'p3' == order[1]