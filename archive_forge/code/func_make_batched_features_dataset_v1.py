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
@tf_export(v1=['data.experimental.make_batched_features_dataset'])
def make_batched_features_dataset_v1(file_pattern, batch_size, features, reader=None, label_key=None, reader_args=None, num_epochs=None, shuffle=True, shuffle_buffer_size=10000, shuffle_seed=None, prefetch_buffer_size=None, reader_num_threads=None, parser_num_threads=None, sloppy_ordering=False, drop_final_batch=False):
    return dataset_ops.DatasetV1Adapter(make_batched_features_dataset_v2(file_pattern, batch_size, features, reader, label_key, reader_args, num_epochs, shuffle, shuffle_buffer_size, shuffle_seed, prefetch_buffer_size, reader_num_threads, parser_num_threads, sloppy_ordering, drop_final_batch))