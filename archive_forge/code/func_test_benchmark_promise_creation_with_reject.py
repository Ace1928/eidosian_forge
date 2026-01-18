from pytest import raises
import time
from promise import Promise, promisify, is_thenable
def test_benchmark_promise_creation_with_reject(benchmark):
    do_resolve = lambda resolve, reject: reject(Exception('Error'))

    def create_promise():
        p = Promise(do_resolve)
        return p
    with raises(Exception) as exc_info:
        result = benchmark(create_promise).get()
    assert str(exc_info.value) == 'Error'