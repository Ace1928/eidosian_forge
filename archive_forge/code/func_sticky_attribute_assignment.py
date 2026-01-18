import collections
import copy
import sys
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base
from tensorflow.python.trackable import layer_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.tracking.sticky_attribute_assignment', v1=[])
def sticky_attribute_assignment(trackable, name, value):
    """Adds dependencies, generally called from __setattr__.

  This behavior is shared between Trackable and Model.

  Respects NoDependency indicators, but otherwise makes trackable objects
  out of common data structures and tracks objects by their attribute names.

  Args:
    trackable: The object to add dependencies to (generally the one having
      an attribute assigned).
    name: The attribute name being assigned.
    value: The value being assigned. Not necessarily a trackable object.

  Returns:
    The value which should be stored in the attribute (unwrapped from a
    NoDependency object if necessary).
  """
    if isinstance(value, NoDependency):
        add_dependency = False
    else:
        add_dependency = True
    value = wrap_or_unwrap(value)
    if not add_dependency:
        return value
    if isinstance(value, base.Trackable):
        trackable._track_trackable(value, name=name, overwrite=True)
    return value