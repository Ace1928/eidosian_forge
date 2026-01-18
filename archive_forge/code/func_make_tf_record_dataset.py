import collections
import csv
import functools
import gzip
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import error_ops
from tensorflow.python.data.experimental.ops import parsing_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import map_op
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.data.util import convert
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util.tf_export import tf_export
def make_tf_record_dataset(file_pattern, batch_size, parser_fn=None, num_epochs=None, shuffle=True, shuffle_buffer_size=None, shuffle_seed=None, prefetch_buffer_size=None, num_parallel_reads=None, num_parallel_parser_calls=None, drop_final_batch=False):
    """Reads and optionally parses TFRecord files into a dataset.

  Provides common functionality such as batching, optional parsing, shuffling,
  and performant defaults.

  Args:
    file_pattern: List of files or patterns of TFRecord file paths.
      See `tf.io.gfile.glob` for pattern rules.
    batch_size: An int representing the number of records to combine
      in a single batch.
    parser_fn: (Optional.) A function accepting string input to parse
      and process the record contents. This function must map records
      to components of a fixed shape, so they may be batched. By
      default, uses the record contents unmodified.
    num_epochs: (Optional.) An int specifying the number of times this
      dataset is repeated.  If None (the default), cycles through the
      dataset forever.
    shuffle: (Optional.) A bool that indicates whether the input
      should be shuffled. Defaults to `True`.
    shuffle_buffer_size: (Optional.) Buffer size to use for
      shuffling. A large buffer size ensures better shuffling, but
      increases memory usage and startup time.
    shuffle_seed: (Optional.) Randomization seed to use for shuffling.
    prefetch_buffer_size: (Optional.) An int specifying the number of
      feature batches to prefetch for performance improvement.
      Defaults to auto-tune. Set to 0 to disable prefetching.
    num_parallel_reads: (Optional.) Number of threads used to read
      records from files. By default or if set to a value >1, the
      results will be interleaved. Defaults to `24`.
    num_parallel_parser_calls: (Optional.) Number of parallel
      records to parse in parallel. Defaults to `batch_size`.
    drop_final_batch: (Optional.) Whether the last batch should be
      dropped in case its size is smaller than `batch_size`; the
      default behavior is not to drop the smaller batch.

  Returns:
    A dataset, where each element matches the output of `parser_fn`
    except it will have an additional leading `batch-size` dimension,
    or a `batch_size`-length 1-D tensor of strings if `parser_fn` is
    unspecified.
  """
    if num_parallel_reads is None:
        num_parallel_reads = 24
    if num_parallel_parser_calls is None:
        num_parallel_parser_calls = batch_size
    if prefetch_buffer_size is None:
        prefetch_buffer_size = dataset_ops.AUTOTUNE
    files = dataset_ops.Dataset.list_files(file_pattern, shuffle=shuffle, seed=shuffle_seed)
    dataset = core_readers.TFRecordDataset(files, num_parallel_reads=num_parallel_reads)
    if shuffle_buffer_size is None:
        shuffle_buffer_size = 10000
    dataset = _maybe_shuffle_and_repeat(dataset, num_epochs, shuffle, shuffle_buffer_size, shuffle_seed)
    drop_final_batch = drop_final_batch or num_epochs is None
    if parser_fn is None:
        dataset = dataset.batch(batch_size, drop_remainder=drop_final_batch)
    else:
        dataset = dataset.map(parser_fn, num_parallel_calls=num_parallel_parser_calls)
        dataset = dataset.batch(batch_size, drop_remainder=drop_final_batch)
    if prefetch_buffer_size == 0:
        return dataset
    else:
        return dataset.prefetch(buffer_size=prefetch_buffer_size)