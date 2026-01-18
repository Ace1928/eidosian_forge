import copy
import weakref
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import trackable_view
from tensorflow.python.trackable import base
from tensorflow.python.util.tf_export import tf_export
@property
def root(self):
    if isinstance(self._root_ref, weakref.ref):
        derefed = self._root_ref()
        assert derefed is not None
        return derefed
    else:
        return self._root_ref