from collections import namedtuple
from collections import OrderedDict
import copy
import datetime
import inspect
import logging
from unittest import mock
import fixtures
from oslo_utils.secretutils import md5
from oslo_utils import versionutils as vutils
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
def object_action(self, context, objinst, objmethod, args, kwargs):
    objinst = self._ser.deserialize_entity(context, self._ser.serialize_entity(context, objinst))
    objmethod = str(objmethod)
    args, kwargs = self._canonicalize_args(context, args, kwargs)
    original = objinst.obj_clone()
    with mock.patch('oslo_versionedobjects.base.VersionedObject.indirection_api', new=None):
        result = getattr(objinst, objmethod)(*args, **kwargs)
    updates = self._get_changes(original, objinst)
    updates['obj_what_changed'] = objinst.obj_what_changed()
    return (updates, result)