import jax
import numpy as np
from keras.src.utils import jax_utils
def distribute_variable(value, layout):
    """Create a distributed variable for JAX.

    Since JAX doesn't have a variable class, this will just return a `jax.Array`
    with the corresponding layout/sharding specified.

    Note that this function should be used in eager context, not in jitted
    function.

    Args:
        value: the initial value of the variable.
        layout: `TensorLayout` for the created variable, or a
            `jax.sharding.Sharding` instance.

    Returns:
        jax.Array which is the distributed variable.
    """
    if not isinstance(layout, jax.sharding.Sharding):
        layout = _to_jax_layout(layout)
    if isinstance(value, (jax.Array, jax.numpy.ndarray)) and value.sharding.is_equivalent_to(layout, ndim=len(value.shape)):
        return value
    if layout.is_fully_addressable:
        return jax.device_put(value, layout)
    else:
        mapping = layout.addressable_devices_indices_map(value.shape)
        local_values = jax.device_put([value[i] for i in mapping.values()], list(mapping.keys()))
        global_value = jax.make_array_from_single_device_arrays(value.shape, layout, local_values)
        return global_value