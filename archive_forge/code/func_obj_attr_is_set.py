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
def obj_attr_is_set(self, attrname):
    """Test object to see if attrname is present.

        Returns True if the named attribute has a value set, or
        False if not. Raises AttributeError if attrname is not
        a valid attribute for this object.
        """
    if attrname not in self.obj_fields:
        raise AttributeError(_("%(objname)s object has no attribute '%(attrname)s'") % {'objname': self.obj_name(), 'attrname': attrname})
    return hasattr(self, _get_attrname(attrname))