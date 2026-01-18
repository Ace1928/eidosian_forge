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
def obj_reset_changes(self, fields=None, recursive=False):
    """Reset the list of fields that have been changed.

        :param fields: List of fields to reset, or "all" if None.
        :param recursive: Call obj_reset_changes(recursive=True) on
                          any sub-objects within the list of fields
                          being reset.

        This is NOT "revert to previous values".

        Specifying fields on recursive resets will only be honored at the top
        level. Everything below the top will reset all.
        """
    if recursive:
        for field in self.obj_get_changes():
            if fields and field not in fields:
                continue
            if not self.obj_attr_is_set(field):
                continue
            value = getattr(self, field)
            if value is None:
                continue
            if isinstance(self.fields[field], obj_fields.ObjectField):
                value.obj_reset_changes(recursive=True)
            elif isinstance(self.fields[field], obj_fields.ListOfObjectsField):
                for thing in value:
                    thing.obj_reset_changes(recursive=True)
    if fields:
        self._changed_fields -= set(fields)
    else:
        self._changed_fields.clear()