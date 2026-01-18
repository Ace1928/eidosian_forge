from boto.sdb.db.property import Property
from boto.sdb.db.key import Key
from boto.sdb.db.query import Query
import boto
from boto.compat import filter
@classmethod
def get_by_key_name(cls, key_names, parent=None):
    raise NotImplementedError('Key Names are not currently supported')