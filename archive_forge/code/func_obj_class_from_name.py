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
def obj_class_from_name(cls, objname, objver):
    """Returns a class from the registry based on a name and version."""
    if objname not in VersionedObjectRegistry.obj_classes():
        (LOG.error('Unable to instantiate unregistered object type %(objtype)s'), dict(objtype=objname))
        raise exception.UnsupportedObjectError(objtype=objname)
    compatible_match = None
    for objclass in VersionedObjectRegistry.obj_classes()[objname]:
        if objclass.VERSION == objver:
            return objclass
        if not compatible_match and vutils.is_compatible(objver, objclass.VERSION):
            compatible_match = objclass
    if compatible_match:
        return compatible_match
    latest_ver = VersionedObjectRegistry.obj_classes()[objname][0].VERSION
    raise exception.IncompatibleObjectVersion(objname=objname, objver=objver, supported=latest_ver)