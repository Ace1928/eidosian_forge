import copy
import weakref
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import trackable_view
from tensorflow.python.trackable import base
from tensorflow.python.util.tf_export import tf_export
def frozen_saveable_objects(self, object_map=None, to_graph=None, call_with_mapped_captures=None):
    """Creates SaveableObjects with the current object graph frozen."""
    return save_util_v1.frozen_saveables_and_savers(self, object_map, to_graph, call_with_mapped_captures)[0]