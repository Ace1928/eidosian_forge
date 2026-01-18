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
def obj_make_compatible_from_manifest(self, primitive, target_version, version_manifest):
    self._obj_version_manifest = version_manifest
    try:
        return self.obj_make_compatible(primitive, target_version)
    finally:
        delattr(self, '_obj_version_manifest')