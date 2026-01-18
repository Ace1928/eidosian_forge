from typing import List, Optional, Tuple
from absl import logging
import numpy as np
from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python import tpu_util
from tensorflow.python.eager import context
from tensorflow.python.framework import device as tf_device
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.create_distributed_mesh', v1=[])
def create_distributed_mesh(mesh_dims: List[Tuple[str, int]], mesh_name: str='', local_devices: Optional[List[str]]=None, device_type: Optional[str]=None, use_xla_spmd: bool=layout.USE_XLA_SPMD) -> layout.Mesh:
    """Creates a distributed mesh.

  This is similar to `create_mesh`, but with a different set of arguments to
  create a mesh that spans evenly across a multi-client DTensor cluster.

  For CPU and GPU meshes, users can choose to use fewer local devices than what
  is available `local_devices`.

  For TPU, only meshes that uses all TPU cores is supported by the DTensor
  runtime.

  Args:
    mesh_dims: A list of (dim_name, dim_size) tuples.
    mesh_name: Name of the created mesh. Defaults to ''.
    local_devices: String representations of devices to use. This is the device
      part of tf.DeviceSpec, e.g. 'CPU:0'. Defaults to all available local
      logical devices.
    device_type: Type of device to build the mesh for. Defaults to 'CPU'.
      Supported values are 'CPU', 'GPU', 'TPU'.6
    use_xla_spmd: Boolean when True, will use XLA SPMD instead of
      DTensor SPMD.

  Returns:
    A mesh that spans evenly across all DTensor clients in the cluster.
  """
    dim_names, shape = zip(*mesh_dims)
    if not accelerator_util.is_initialized():
        raise ValueError('Accelerators are uninitialized, please run dtensor.initialize_accelerator_system() first.')
    if device_type and device_type.upper() == 'TPU':
        if local_devices is not None:
            raise ValueError(f'Do not specify devices for {device_type.upper()} meshes. Using a partial list of devices for {device_type.upper()} is not supported.')
    device_specs, device_type = _make_device_specs(local_devices, device_type)
    if device_type.upper() in ['CPU', 'GPU']:
        local_spec = tf_device.DeviceSpec(job=config.job_name(), replica=0, task=config.client_id())
        device_specs = [local_spec.make_merged_spec(d) for d in device_specs]
        num_global_devices = len(device_specs) * config.num_clients()
        if np.prod(shape) != num_global_devices:
            raise ValueError(f'Global number of devices ({len(device_specs)} per client * {config.num_clients()} clients = {num_global_devices}) must be equal to total size of the mesh of shape {shape}')
        global_device_ids = np.arange(num_global_devices).reshape(shape)
        flattened = np.ravel(global_device_ids).tolist()
        start_idx = len(device_specs) * config.client_id()
        local_device_ids = flattened[start_idx:start_idx + len(device_specs)]
        mesh = layout.Mesh(dim_names=dim_names, global_device_ids=global_device_ids, local_device_ids=local_device_ids, local_devices=device_specs, mesh_name=mesh_name, use_xla_spmd=use_xla_spmd)
        _print_context(num_global_devices, config.num_clients(), config.client_id(), device_type, mesh)
        return mesh
    if device_type.upper() == 'TPU':
        mesh = tpu_util.create_tpu_mesh(mesh_dim_names=dim_names, mesh_shape=shape, mesh_name=mesh_name, use_xla_spmd=use_xla_spmd)
        _print_context(config.num_global_devices(device_type), config.num_clients(), config.client_id(), device_type, mesh)
        return mesh
    raise ValueError(f'Device type {device_type} is not CPU, GPU or TPU')