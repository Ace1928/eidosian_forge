from concurrent.futures import ThreadPoolExecutor
from promise import Promise
import time
import weakref
import gc
def test_issue_33():

    def do(x):
        v = Promise.resolve('ok').then(lambda x: x).get()
        return v
    p = Promise.resolve(None).then(do)
    assert p.get() == 'ok'