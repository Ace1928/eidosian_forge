import copy
import weakref
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import trackable_view
from tensorflow.python.trackable import base
from tensorflow.python.util.tf_export import tf_export
def list_children(self, obj, save_type=base.SaveType.CHECKPOINT, **kwargs):
    """Returns list of all child trackables attached to obj.

    Args:
      obj: A `Trackable` object.
      save_type: A string, can be 'savedmodel' or 'checkpoint'.
      **kwargs: kwargs to use when retrieving the object's children.

    Returns:
      List of all children attached to the object.
    """
    children = []
    for name, ref in super(ObjectGraphView, self).children(obj, save_type, **kwargs).items():
        children.append(base.TrackableReference(name, ref))
    if obj is self.root and self._attached_dependencies:
        children.extend(self._attached_dependencies)
    return children