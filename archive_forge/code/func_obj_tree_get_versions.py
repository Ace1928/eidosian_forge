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
def obj_tree_get_versions(objname, tree=None):
    """Construct a mapping of dependent object versions.

    This method builds a list of dependent object versions given a top-
    level object with other objects as fields. It walks the tree recursively
    to determine all the objects (by symbolic name) that could be contained
    within the top-level object, and the maximum versions of each. The result
    is a dict like::

      {'MyObject': '1.23', ... }

    :param objname: The top-level object at which to start
    :param tree: Used internally, pass None here.
    :returns: A dictionary of object names and versions
    """
    if tree is None:
        tree = {}
    if objname in tree:
        return tree
    objclass = VersionedObjectRegistry.obj_classes()[objname][0]
    tree[objname] = objclass.VERSION
    for field_name in objclass.fields:
        field = objclass.fields[field_name]
        if isinstance(field, obj_fields.ObjectField):
            child_cls = field._type._obj_name
        elif isinstance(field, obj_fields.ListOfObjectsField):
            child_cls = field._type._element_type._type._obj_name
        else:
            continue
        try:
            obj_tree_get_versions(child_cls, tree=tree)
        except IndexError:
            raise exception.UnregisteredSubobject(child_objname=child_cls, parent_objname=objname)
    return tree