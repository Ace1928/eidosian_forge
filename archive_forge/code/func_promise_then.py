from time import sleep
from concurrent.futures import ThreadPoolExecutor
from promise import Promise
from operator import mul
def promise_then(r):
    return mul(r, n)