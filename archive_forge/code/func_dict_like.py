from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml_location_value
from googlecloudsdk.core.util import files
from ruamel import yaml
import six
def dict_like(item):
    """Return True if the item is like a dict: a MutableMapping."""
    return isinstance(item, collections_abc.MutableMapping)