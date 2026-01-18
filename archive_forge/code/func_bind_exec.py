import logging
from collections import OrderedDict
from .. import context as ctx
from .. import ndarray as nd
from ..io import DataDesc
from ..executor_manager import _split_input_slice
from ..ndarray import _DTYPE_MX_TO_NP
def bind_exec(self, data_shapes, label_shapes, shared_group=None, reshape=False):
    """Bind executors on their respective devices.

        Parameters
        ----------
        data_shapes : list
        label_shapes : list
        shared_group : DataParallelExecutorGroup
        reshape : bool
        """
    assert reshape or not self.execs
    self.batch_size = None
    self.data_layouts = self.decide_slices(data_shapes)
    if label_shapes is not None:
        self.label_layouts = self.decide_slices(label_shapes)
    for i in range(len(self.contexts)):
        data_shapes_i = self._sliced_shape(data_shapes, i, self.data_layouts)
        if label_shapes is not None:
            label_shapes_i = self._sliced_shape(label_shapes, i, self.label_layouts)
        else:
            label_shapes_i = []
        if reshape:
            self.execs[i] = self._default_execs[i].reshape(allow_up_sizing=True, **dict(data_shapes_i + label_shapes_i))
        else:
            self.execs.append(self._bind_ith_exec(i, data_shapes_i, label_shapes_i, shared_group))
    self.data_shapes = data_shapes
    self.label_shapes = label_shapes
    self.data_names = [i.name for i in self.data_shapes]
    if label_shapes is not None:
        self.label_names = [i.name for i in self.label_shapes]
    self._collect_arrays()