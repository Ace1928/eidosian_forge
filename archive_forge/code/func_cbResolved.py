from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python.failure import Failure
def cbResolved(results):
    answers, authority, additional = results
    answers.insert(0, previous)
    return (answers, authority, additional)