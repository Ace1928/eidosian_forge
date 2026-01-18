from promise import Promise
from .utils import assert_exception
from threading import Event
def test_chained_promises():
    """
    Handles the case where the arguments to then
    are values, not functions or promises.
    """
    p1 = Promise(lambda resolve, reject: resolve(Promise.resolve(True)))
    assert p1.get() == True