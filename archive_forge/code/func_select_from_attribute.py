import collections
import functools
from oslo_utils import strutils
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import support
from oslo_log import log as logging
def select_from_attribute(attribute_value, path):
    """Select an element from an attribute value.

    :param attribute_value: the attribute value.
    :param path: a list of path components to select from the attribute.
    :returns: the selected attribute component value.
    """

    def get_path_component(collection, key):
        if not isinstance(collection, (collections.abc.Mapping, collections.abc.Sequence)):
            raise TypeError(_("Can't traverse attribute path"))
        if not isinstance(key, (str, int)):
            raise TypeError(_('Path components in attributes must be strings'))
        return collection[key]
    try:
        return functools.reduce(get_path_component, path, attribute_value)
    except (KeyError, IndexError, TypeError):
        return None