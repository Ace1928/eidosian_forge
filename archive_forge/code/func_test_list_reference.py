import logging
import time
from datetime import datetime
from boto.sdb.db.model import Model
from boto.sdb.db.property import StringProperty, IntegerProperty, BooleanProperty
from boto.sdb.db.property import DateTimeProperty, FloatProperty, ReferenceProperty
from boto.sdb.db.property import PasswordProperty, ListProperty, MapProperty
from boto.exception import SDBPersistenceError
def test_list_reference():
    global _objects
    t = TestBasic()
    t.put()
    _objects['test_list_ref_t'] = t
    tt = TestListReference()
    tt.name = 'foo'
    tt.basics = [t]
    tt.put()
    time.sleep(5)
    _objects['test_list_ref_tt'] = tt
    ttt = TestListReference.get_by_id(tt.id)
    assert ttt.basics[0].id == t.id