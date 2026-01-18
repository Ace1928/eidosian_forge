from pytest import raises
from promise import Promise
from promise.promise_list import PromiseList
def test_bad_promises():
    all_promises = all(None)
    with raises(Exception) as exc_info:
        all_promises.get()
    assert str(exc_info.value) == 'PromiseList requires an iterable. Received None.'