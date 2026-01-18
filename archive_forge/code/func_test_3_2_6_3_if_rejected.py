from promise import Promise
from .utils import assert_exception
from threading import Event
def test_3_2_6_3_if_rejected():
    """
    Testing return of pending promises to make
    sure they are properly chained.
    This covers the case where the root promise
    is rejected before the chaining is defined.
    """
    p1 = Promise()
    p1.do_reject(Exception('Error'))
    pending = Promise()
    pending.do_resolve(10)
    pr = p1.then(None, lambda r: pending)
    pending._wait()
    assert pending.is_fulfilled
    assert 10 == pending.get()
    pr._wait()
    assert pr.is_fulfilled
    assert 10 == pr.get()
    p2 = Promise()
    p2.do_reject(Exception('Error'))
    bad = Promise()
    bad.do_reject(Exception('Assertion'))
    pr = p2.then(None, lambda r: bad)
    bad._wait()
    assert bad.is_rejected
    assert_exception(bad.reason, Exception, 'Assertion')
    pr._wait()
    assert pr.is_rejected
    assert_exception(pr.reason, Exception, 'Assertion')