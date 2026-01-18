from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.summary.writer import writer
from tensorflow.python.summary.writer import writer_cache
def add_meta_graph(self, meta_graph_def, global_step=None):
    """Add metagraph."""
    if global_step is not None and global_step < 0:
        raise ValueError('Invalid global_step %s.' % global_step)
    self._added_meta_graphs.append(meta_graph_def)