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
def obj_to_primitive(self, target_version=None, version_manifest=None):
    """Simple base-case dehydration.

        This calls to_primitive() for each item in fields.
        """
    if target_version is None:
        target_version = self.VERSION
    if vutils.convert_version_to_tuple(target_version) > vutils.convert_version_to_tuple(self.VERSION):
        raise exception.InvalidTargetVersion(version=target_version)
    primitive = dict()
    for name, field in self.fields.items():
        if self.obj_attr_is_set(name):
            primitive[name] = field.to_primitive(self, name, getattr(self, name))
    if target_version != self.VERSION or version_manifest:
        self.obj_make_compatible_from_manifest(primitive, target_version, version_manifest)
    obj = {self._obj_primitive_key('name'): self.obj_name(), self._obj_primitive_key('namespace'): self.OBJ_PROJECT_NAMESPACE, self._obj_primitive_key('version'): target_version, self._obj_primitive_key('data'): primitive}
    if self.obj_what_changed():
        what_changed = self.obj_what_changed()
        changes = [field for field in what_changed if field in primitive]
        if changes:
            obj[self._obj_primitive_key('changes')] = changes
    return obj