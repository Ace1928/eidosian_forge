from pytest import raises
import time
from promise import Promise, promisify, is_thenable
def test_benchmark_is_thenable_basic_type(benchmark):

    def create_promise():
        return is_thenable(True)
    result = benchmark(create_promise)
    assert result == False