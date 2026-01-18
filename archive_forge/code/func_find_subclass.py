from boto.sdb.db.property import Property
from boto.sdb.db.key import Key
from boto.sdb.db.query import Query
import boto
from boto.compat import filter
@classmethod
def find_subclass(cls, name):
    """Find a subclass with a given name"""
    if name == cls.__name__:
        return cls
    for sc in cls.__sub_classes__:
        r = sc.find_subclass(name)
        if r is not None:
            return r