import copy
import warnings
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.legacy_tf_layers import variable_scope_shim
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
def add_weight(self, name, shape, dtype=None, initializer=None, regularizer=None, trainable=None, constraint=None, use_resource=None, synchronization=vs.VariableSynchronization.AUTO, aggregation=vs.VariableAggregation.NONE, partitioner=None, **kwargs):
    """Adds a new variable to the layer, or gets an existing one; returns it.

    Args:
      name: variable name.
      shape: variable shape.
      dtype: The type of the variable. Defaults to `self.dtype` or `float32`.
      initializer: initializer instance (callable).
      regularizer: regularizer instance (callable).
      trainable: whether the variable should be part of the layer's
        "trainable_variables" (e.g. variables, biases)
        or "non_trainable_variables" (e.g. BatchNorm mean, stddev).
        Note, if the current variable scope is marked as non-trainable
        then this parameter is ignored and any added variables are also
        marked as non-trainable. `trainable` defaults to `True` unless
        `synchronization` is set to `ON_READ`.
      constraint: constraint instance (callable).
      use_resource: Whether to use `ResourceVariable`.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses
        when to synchronize. If `synchronization` is set to `ON_READ`,
        `trainable` must not be set to `True`.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      partitioner: (optional) partitioner instance (callable).  If
        provided, when the requested variable is created it will be split
        into multiple partitions according to `partitioner`.  In this case,
        an instance of `PartitionedVariable` is returned.  Available
        partitioners include `tf.compat.v1.fixed_size_partitioner` and
        `tf.compat.v1.variable_axis_size_partitioner`.  For more details, see
        the documentation of `tf.compat.v1.get_variable` and the  "Variable
        Partitioners and Sharding" section of the API guide.
      **kwargs: Additional keyword arguments.

    Returns:
      The created variable.  Usually either a `Variable` or `ResourceVariable`
      instance.  If `partitioner` is not `None`, a `PartitionedVariable`
      instance is returned.

    Raises:
      RuntimeError: If called with partitioned variable regularization and
        eager execution is enabled.
      ValueError: When trainable has been set to True with synchronization
        set as `ON_READ`.
    """
    for kwarg in kwargs:
        if kwarg != 'experimental_autocast':
            raise TypeError('Unknown keyword argument:', kwarg)
    if self._keras_style:
        return super(Layer, self).add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, trainable=trainable and self.trainable, constraint=constraint, use_resource=use_resource, synchronization=vs.VariableSynchronization.AUTO, aggregation=vs.VariableAggregation.NONE, partitioner=partitioner, **kwargs)
    if synchronization == vs.VariableSynchronization.ON_READ:
        if trainable:
            raise ValueError('Synchronization value can be set to VariableSynchronization.ON_READ only for non-trainable variables. You have specified trainable=True and synchronization=VariableSynchronization.ON_READ.')
        else:
            trainable = False
    elif trainable is None:
        trainable = True

    def _should_add_regularizer(variable, existing_variable_set):
        if base_layer_utils.is_split_variable(variable):
            for var in variable:
                if var in existing_variable_set:
                    return False
            return True
        else:
            return variable not in existing_variable_set
    init_graph = None
    if not context.executing_eagerly():
        default_graph = ops.get_default_graph()
        if default_graph.building_function:
            with ops.init_scope():
                if not context.executing_eagerly():
                    init_graph = ops.get_default_graph()
                    existing_variables = set(tf_variables.global_variables())
        else:
            init_graph = default_graph
            existing_variables = set(tf_variables.global_variables())
    if dtype is None:
        dtype = self.dtype or dtypes.float32
    self._set_scope(None)
    reuse = self.built or self._reuse
    prev_len_trainable = len(self._trainable_weights)
    with vs.variable_scope(self._scope, reuse=reuse, auxiliary_name_scope=False) as scope:
        self._current_scope = scope
        with backend.name_scope(self._name_scope()):
            use_resource = use_resource or self._use_resource_variables or scope.use_resource
            if initializer is None:
                initializer = scope.initializer
            variable = super(Layer, self).add_weight(name, shape, dtype=dtypes.as_dtype(dtype), initializer=initializer, trainable=trainable and self.trainable, constraint=constraint, partitioner=partitioner, use_resource=use_resource, synchronization=synchronization, aggregation=aggregation, getter=vs.get_variable, **kwargs)
            if regularizer:
                if ops.executing_eagerly_outside_functions() or _should_add_regularizer(variable, existing_variables):
                    self._handle_weight_regularization(name, variable, regularizer)
                    var_store = vs._get_default_variable_store()
                    if hasattr(var_store, 'add_regularizer'):
                        var_store.add_regularizer(variable, regularizer)
            if init_graph is not None:
                with init_graph.as_default():
                    trainable_variables = tf_variables.trainable_variables()
                if trainable and self.trainable and (variable not in trainable_variables):
                    extra_trainable_vars = self._trainable_weights[prev_len_trainable:]
                    self._trainable_weights = self._trainable_weights[:prev_len_trainable]
                    self._non_trainable_weights += extra_trainable_vars
    return variable