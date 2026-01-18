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
def test_class_action_falls_back_compat(self):
    with mock.patch.object(base.VersionedObject, 'indirection_api') as ma:
        ma.object_class_action_versions.side_effect = NotImplementedError
        MyObj.query(self.context)
        ma.object_class_action.assert_called_once_with(self.context, 'MyObj', 'query', MyObj.VERSION, (), {})