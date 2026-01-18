from promise import Promise
from .utils import assert_exception
from threading import Event
def test_3_2_6_2_if():
    """
    Promises returned by then must be rejected when any of their
    callbacks throw an exception.
    """

    def fail(v):
        raise AssertionError('Exception Message')
    p1 = Promise.resolve(5)
    pf = p1.then(fail)
    pf._wait()
    assert pf.is_rejected
    assert_exception(pf.reason, AssertionError, 'Exception Message')
    p2 = Promise.reject(Exception('Error'))
    pr = p2.then(None, fail)
    pr._wait()
    assert pr.is_rejected
    assert_exception(pr.reason, AssertionError, 'Exception Message')