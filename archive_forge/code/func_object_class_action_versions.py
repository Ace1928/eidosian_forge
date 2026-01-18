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
def object_class_action_versions(self, context, objname, objmethod, object_versions, args, kwargs):
    objname = str(objname)
    objmethod = str(objmethod)
    object_versions = {str(o): str(v) for o, v in object_versions.items()}
    args, kwargs = self._canonicalize_args(context, args, kwargs)
    objver = object_versions[objname]
    cls = base.VersionedObject.obj_class_from_name(objname, objver)
    with mock.patch('oslo_versionedobjects.base.VersionedObject.indirection_api', new=None):
        result = getattr(cls, objmethod)(context, *args, **kwargs)
    return base.VersionedObject.obj_from_primitive(result.obj_to_primitive(target_version=objver), context=context) if isinstance(result, base.VersionedObject) else result