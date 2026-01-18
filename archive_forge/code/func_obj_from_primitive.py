import abc
import collections
from collections import abc as collections_abc
import copy
import functools
import logging
import warnings
import oslo_messaging as messaging
from oslo_utils import excutils
from oslo_utils import versionutils as vutils
from oslo_versionedobjects._i18n import _
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields as obj_fields
@classmethod
def obj_from_primitive(cls, primitive, context=None):
    """Object field-by-field hydration."""
    objns = cls._obj_primitive_field(primitive, 'namespace')
    objname = cls._obj_primitive_field(primitive, 'name')
    objver = cls._obj_primitive_field(primitive, 'version')
    if objns != cls.OBJ_PROJECT_NAMESPACE:
        raise exception.UnsupportedObjectError(objtype='%s.%s' % (objns, objname))
    objclass = cls.obj_class_from_name(objname, objver)
    return objclass._obj_from_primitive(context, objver, primitive)