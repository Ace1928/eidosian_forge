from asyncio import coroutine
from pytest import mark
from time import sleep
from promise import Promise
def resolve_or_reject(resolve, reject):
    sleep(0.1)
    resolve(True)