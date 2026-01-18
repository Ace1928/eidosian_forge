import functools
from tensorflow.python.eager import context
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import function_deserialization
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import signature_serialization
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import resource
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import nest
def restore_variables(self, wrapped, restore_from_saver):
    """Restores variables from the checkpoint."""
    if restore_from_saver is not None:
        initializer, _ = restore_from_saver(constant_op.constant(self._variables_path))
        if not ops.executing_eagerly_outside_functions():
            ops.add_to_collection('saved_model_initializers', initializer)
            one_unlifted = False
            for variable in wrapped.graph.get_collection_ref(ops.GraphKeys.GLOBAL_VARIABLES):
                if variable.graph is wrapped.graph:
                    one_unlifted = True
                variable._initializer_op = initializer
            if one_unlifted:
                logging.warning('Some variables could not be lifted out of a loaded function. Please run `sess.run(tf.get_collection("saved_model_initializers"))`to restore these variables.')