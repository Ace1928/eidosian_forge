import copy
import datetime
import jsonschema
import logging
import pytz
from unittest import mock
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import testtools
from testtools import matchers
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_list_object_concat_fails_different_objects(self):

    @base.VersionedObjectRegistry.register_if(False)
    class MyList(base.ObjectListBase, base.VersionedObject):
        fields = {'objects': fields.ListOfObjectsField('MyOwnedObject')}

    @base.VersionedObjectRegistry.register_if(False)
    class MyList2(base.ObjectListBase, base.VersionedObject):
        fields = {'objects': fields.ListOfObjectsField('MyOwnedObject')}
    list1 = MyList(objects=[MyOwnedObject(baz=1)])
    list2 = MyList2(objects=[MyOwnedObject(baz=2)])

    def add(x, y):
        return x + y
    self.assertRaises(TypeError, add, list1, list2)
    self.assertEqual(1, len(list1.objects))
    self.assertEqual(1, len(list2.objects))
    self.assertEqual(1, list1.objects[0].baz)
    self.assertEqual(2, list2.objects[0].baz)