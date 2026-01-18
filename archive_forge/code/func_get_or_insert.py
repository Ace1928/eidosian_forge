from boto.sdb.db.property import Property
from boto.sdb.db.key import Key
from boto.sdb.db.query import Query
import boto
from boto.compat import filter
@classmethod
def get_or_insert(key_name, **kw):
    raise NotImplementedError('get_or_insert not currently supported')