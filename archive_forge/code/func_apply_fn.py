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
def apply_fn(dataset):
    return core_readers.ParallelInterleaveDataset(dataset, lambda filename: reader(filename, *reader_args), cycle_length=reader_num_threads, block_length=1, sloppy=sloppy_ordering, buffer_output_elements=None, prefetch_input_elements=None)