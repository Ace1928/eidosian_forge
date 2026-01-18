import collections
import copy
import datetime
import hashlib
import inspect
from unittest import mock
import iso8601
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_get_fingerprint_with_extra_data(self):

    class ExtraDataObj(base.VersionedObject):
        pass

    def get_data(obj_class):
        return (obj_class,)
    ExtraDataObj.VERSION = '1.1'
    argspec = 'cubone'
    self._add_class(self.obj_classes, ExtraDataObj)
    with mock.patch.object(fixture, 'get_method_spec') as mock_gas:
        mock_gas.return_value = argspec
        fp = self.ovc._get_fingerprint(ExtraDataObj.__name__, extra_data_func=get_data)
    exp_fields = []
    exp_methods = []
    exp_extra_data = ExtraDataObj
    exp_relevant_data = (exp_fields, exp_methods, exp_extra_data)
    expected_hash = hashlib.md5(bytes(repr(exp_relevant_data).encode())).hexdigest()
    expected_fp = '%s-%s' % (ExtraDataObj.VERSION, expected_hash)
    self.assertEqual(expected_fp, fp, '_get_fingerprint() did not generate a correct fingerprint.')