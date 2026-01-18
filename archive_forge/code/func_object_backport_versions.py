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
def object_backport_versions(self, context, objinst, object_versions):
    """Perform a backport of an object instance.

        This method is basically just like object_backport() but instead of
        providing a specific target version for the toplevel object and
        relying on the service-side mapping to handle sub-objects, this sends
        a mapping of all the dependent objects and their client-supported
        versions. The server will backport objects within the tree starting
        at objinst to the versions specified in object_versions, removing
        objects that have no entry. Use obj_tree_get_versions() to generate
        this mapping.

        NOTE: This was not in the initial spec for this interface, so the
        base class raises NotImplementedError if you don't implement it.
        For backports, this method will be tried first, and if unimplemented,
        will fall back to object_backport().

        :param context: The context within which to perform the backport
        :param objinst: An instance of a VersionedObject to be backported
        :param object_versions: A dict of {objname: version} mappings
        """
    warnings.warn('object_backport() is deprecated in favor of object_backport_versions() and will be removed in a later release', DeprecationWarning)
    raise NotImplementedError('Multi-version backport not supported')