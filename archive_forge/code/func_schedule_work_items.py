import os
from typing import Union, Optional, Tuple, Any, List, Sized, TypeVar
import itertools
from collections import namedtuple
import parlai.utils.logging as logging
import torch.optim
@staticmethod
def schedule_work_items(layers: torch.nn.ModuleList, chunks: List[Chunk]):
    """
        Iterate through chunks and layers that should be pipelined.

        Each iteration of this generator yields the following properties:

            - layer_nos: a list of indices of layers for you to forward through
            - chunk_idx: the index of the chunk we are manipulating. Use this
              if you need to update chunk representations.
            - next_device: where the chunk should be moved to AFTER the layer
              computation is done.
        """
    num_chunks = len(chunks)
    for l in layers:
        if not hasattr(l, '_mp_gpu'):
            raise RuntimeError('You must run PipelineHelper.make_parallel on the ModuleList before you can use iterate_layers_chunks.')
    devices = {device_idx: (dev, list(grp)) for device_idx, (dev, grp) in enumerate(itertools.groupby(range(len(layers)), lambda x: layers[x]._mp_gpu))}
    num_timesteps = len(devices) + num_chunks
    for timestep in range(num_timesteps):
        for chunk_idx in range(num_chunks):
            device_idx = timestep - chunk_idx
            if device_idx >= 0 and device_idx < len(devices):
                dev, layers_nos = devices[device_idx]
                next_device, _ = devices[(device_idx + 1) % len(devices)]
                assert device_idx in devices
                yield PipelineWorkItem(chunk_idx=chunk_idx, layer_nos=layers_nos, next_device=next_device)