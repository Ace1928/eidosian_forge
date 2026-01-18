import jax
import numpy as np
from keras.src.utils import jax_utils
def distribute_data_input(inputs, layout):
    """Distribute the input data with the corresponding layout.

    Note that the inputs here is a local worker batch. Within the local worker,
    the data need to be further partitioned to map to the each of the devices.

    Args:
        inputs: `jax.Array` that is already sharded to a local process size.
        layout: `TensorLayout` for the distribution information, or a
            `jax.sharding.Sharding` instance.

    Returns:
        Distributed inputs thats been properly put to local devices.
    """
    if not isinstance(layout, jax.sharding.Sharding):
        layout = _to_jax_layout(layout)
    if layout.is_fully_addressable:
        return jax.device_put(inputs, layout)
    jax_mesh = layout.mesh
    mesh_rank = len(jax_mesh.shape)
    per_process_batch_size = inputs.shape[0]
    if mesh_rank == 1:
        num_split = jax.local_device_count()
        per_replica_batch_size = per_process_batch_size // num_split
        if per_process_batch_size % per_replica_batch_size != 0:
            raise ValueError(f'The local batch size {per_process_batch_size} is notdivisible by the number of local replicas {num_split}')
        global_batch_size = per_process_batch_size * jax.process_count()
        per_replica_batches = jax.numpy.split(inputs, num_split, axis=0)
    elif mesh_rank == 2:
        mesh_batch_dim_size = list(jax_mesh.shape.values())[0]
        local_device_count = jax.local_device_count()
        if mesh_batch_dim_size < local_device_count:
            global_batch_size = per_process_batch_size
            per_replica_batches = [inputs for _ in range(local_device_count)]
        else:
            global_batch_size = per_process_batch_size * (mesh_batch_dim_size // local_device_count)
            per_replica_batches = jax.numpy.split(inputs, local_device_count, axis=0)
    else:
        raise ValueError(f'Only 1D or 2D mesh is supported at the moment. Received mesh shape = {jax_mesh.shape}')
    global_shape = (global_batch_size,) + inputs.shape[1:]
    global_batch_array = jax.make_array_from_single_device_arrays(global_shape, layout, arrays=[jax.device_put(batch, device) for batch, device in zip(per_replica_batches, layout.addressable_devices)])
    return global_batch_array