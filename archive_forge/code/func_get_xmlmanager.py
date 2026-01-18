from boto.sdb.db.property import Property
from boto.sdb.db.key import Key
from boto.sdb.db.query import Query
import boto
from boto.compat import filter
@classmethod
def get_xmlmanager(cls):
    if not hasattr(cls, '_xmlmanager'):
        from boto.sdb.db.manager.xmlmanager import XMLManager
        cls._xmlmanager = XMLManager(cls, None, None, None, None, None, None, None, False)
    return cls._xmlmanager