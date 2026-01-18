from pytest import raises
import time
from promise import Promise, promisify, is_thenable
def create_promise():
    return Promise.all(values)