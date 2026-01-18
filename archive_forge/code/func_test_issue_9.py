from concurrent.futures import ThreadPoolExecutor
from promise import Promise
import time
import weakref
import gc
def test_issue_9():
    no_wait = Promise.all([promise_with_wait(x1, None).then(lambda y: x1 * y) for x1 in (0, 1, 2, 3)]).get()
    wait_a_bit = Promise.all([promise_with_wait(x2, 0.05).then(lambda y: x2 * y) for x2 in (0, 1, 2, 3)]).get()
    wait_longer = Promise.all([promise_with_wait(x3, 0.1).then(lambda y: x3 * y) for x3 in (0, 1, 2, 3)]).get()
    assert no_wait == wait_a_bit
    assert no_wait == wait_longer