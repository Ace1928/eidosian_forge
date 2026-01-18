from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python.failure import Failure
def findAnswerOrCName(name, type, cls):
    cname = None
    for record in records.get(name, []):
        if record.cls == cls:
            if record.type == type:
                return record
            elif record.type == dns.CNAME:
                cname = record
    return cname