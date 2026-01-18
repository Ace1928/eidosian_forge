import collections
import copy
import functools
import itertools
import operator
from heat.common import exception
from heat.engine import function
from heat.engine import properties
def render_hot(self):
    """Return a HOT snippet for the resource definition."""
    if self._rendering is None:
        attrs = {'type': 'resource_type', 'properties': '_properties', 'metadata': '_metadata', 'deletion_policy': '_deletion_policy', 'update_policy': '_update_policy', 'depends_on': '_depends', 'external_id': '_external_id', 'condition': '_condition'}

        def rawattrs():
            """Get an attribute with function objects stripped out."""
            for key, attr in attrs.items():
                value = getattr(self, attr)
                if value is not None:
                    yield (key, copy.deepcopy(value))
        self._rendering = dict(rawattrs())
    return self._rendering