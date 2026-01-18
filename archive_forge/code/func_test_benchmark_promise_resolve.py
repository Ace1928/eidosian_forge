from pytest import raises
import time
from promise import Promise, promisify, is_thenable
def test_benchmark_promise_resolve(benchmark):

    def create_promise():
        return Promise.resolve(True)
    result = benchmark(create_promise).get()
    assert result == True