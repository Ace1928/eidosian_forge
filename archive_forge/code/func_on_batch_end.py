import os
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.summary import summary as tf_summary
from tensorflow.python.training import saver
def on_batch_end(self, batch, logs=None):
    """Writes scalar summaries for metrics on every training batch.

    Performs profiling if current batch is in profiler_batches.
    """
    logs = logs or {}
    self._samples_seen += logs.get('size', 1)
    samples_seen_since = self._samples_seen - self._samples_seen_at_last_write
    if self.update_freq != 'epoch' and samples_seen_since >= self.update_freq:
        batch_logs = {'batch_' + k: v for k, v in logs.items() if k not in ['batch', 'size', 'num_steps']}
        self._write_custom_summaries(self._total_batches_seen, batch_logs)
        self._samples_seen_at_last_write = self._samples_seen
    self._total_batches_seen += 1
    self._stop_profiler()