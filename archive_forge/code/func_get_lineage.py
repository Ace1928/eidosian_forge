from boto.sdb.db.property import Property
from boto.sdb.db.key import Key
from boto.sdb.db.query import Query
import boto
from boto.compat import filter
@classmethod
def get_lineage(cls):
    l = [c.__name__ for c in cls.mro()]
    l.reverse()
    return '.'.join(l)