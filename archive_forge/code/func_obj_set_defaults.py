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
def obj_set_defaults(self, *attrs):
    if not attrs:
        attrs = [name for name, field in self.fields.items() if field.default != obj_fields.UnspecifiedDefault]
    for attr in attrs:
        default = copy.deepcopy(self.fields[attr].default)
        if default is obj_fields.UnspecifiedDefault:
            raise exception.ObjectActionError(action='set_defaults', reason='No default set for field %s' % attr)
        if not self.obj_attr_is_set(attr):
            setattr(self, attr, default)