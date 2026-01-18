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
def test_fake_indirection_serializes_arguments(self):
    ser = mock.MagicMock()
    iapi = fixture.FakeIndirectionAPI(serializer=ser)
    arg1 = mock.MagicMock()
    arg2 = mock.MagicMock()
    iapi.object_action(mock.sentinel.context, mock.sentinel.objinst, mock.sentinel.objmethod, (arg1,), {'foo': arg2})
    ser.serialize_entity.assert_any_call(mock.sentinel.context, arg1)
    ser.serialize_entity.assert_any_call(mock.sentinel.context, arg2)