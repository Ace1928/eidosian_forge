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
@tf_export('data.experimental.make_csv_dataset', v1=[])
def make_csv_dataset_v2(file_pattern, batch_size, column_names=None, column_defaults=None, label_name=None, select_columns=None, field_delim=',', use_quote_delim=True, na_value='', header=True, num_epochs=None, shuffle=True, shuffle_buffer_size=10000, shuffle_seed=None, prefetch_buffer_size=None, num_parallel_reads=None, sloppy=False, num_rows_for_inference=100, compression_type=None, ignore_errors=False, encoding='utf-8'):
    """Reads CSV files into a dataset.

  Reads CSV files into a dataset, where each element of the dataset is a
  (features, labels) tuple that corresponds to a batch of CSV rows. The features
  dictionary maps feature column names to `Tensor`s containing the corresponding
  feature data, and labels is a `Tensor` containing the batch's label data.

  By default, the first rows of the CSV files are expected to be headers listing
  the column names. If the first rows are not headers, set `header=False` and
  provide the column names with the `column_names` argument.

  By default, the dataset is repeated indefinitely, reshuffling the order each
  time. This behavior can be modified by setting the `num_epochs` and `shuffle`
  arguments.

  For example, suppose you have a CSV file containing

  | Feature_A | Feature_B |
  | --------- | --------- |
  | 1         | "a"       |
  | 2         | "b"       |
  | 3         | "c"       |
  | 4         | "d"       |

  ```
  # No label column specified
  dataset = tf.data.experimental.make_csv_dataset(filename, batch_size=2)
  iterator = dataset.as_numpy_iterator()
  print(dict(next(iterator)))
  # prints a dictionary of batched features:
  # OrderedDict([('Feature_A', array([1, 4], dtype=int32)),
  #              ('Feature_B', array([b'a', b'd'], dtype=object))])
  ```

  ```
  # Set Feature_B as label column
  dataset = tf.data.experimental.make_csv_dataset(
      filename, batch_size=2, label_name="Feature_B")
  iterator = dataset.as_numpy_iterator()
  print(next(iterator))
  # prints (features, labels) tuple:
  # (OrderedDict([('Feature_A', array([1, 2], dtype=int32))]),
  #  array([b'a', b'b'], dtype=object))
  ```

  See the
  [Load CSV data guide](https://www.tensorflow.org/tutorials/load_data/csv) for
  more examples of using `make_csv_dataset` to read CSV data.

  Args:
    file_pattern: List of files or patterns of file paths containing CSV
      records. See `tf.io.gfile.glob` for pattern rules.
    batch_size: An int representing the number of records to combine
      in a single batch.
    column_names: An optional list of strings that corresponds to the CSV
      columns, in order. One per column of the input record. If this is not
      provided, infers the column names from the first row of the records.
      These names will be the keys of the features dict of each dataset element.
    column_defaults: A optional list of default values for the CSV fields. One
      item per selected column of the input record. Each item in the list is
      either a valid CSV dtype (float32, float64, int32, int64, or string), or a
      `Tensor` with one of the aforementioned types. The tensor can either be
      a scalar default value (if the column is optional), or an empty tensor (if
      the column is required). If a dtype is provided instead of a tensor, the
      column is also treated as required. If this list is not provided, tries
      to infer types based on reading the first num_rows_for_inference rows of
      files specified, and assumes all columns are optional, defaulting to `0`
      for numeric values and `""` for string values. If both this and
      `select_columns` are specified, these must have the same lengths, and
      `column_defaults` is assumed to be sorted in order of increasing column
      index.
    label_name: A optional string corresponding to the label column. If
      provided, the data for this column is returned as a separate `Tensor` from
      the features dictionary, so that the dataset complies with the format
      expected by a `tf.Estimator.train` or `tf.Estimator.evaluate` input
      function.
    select_columns: An optional list of integer indices or string column
      names, that specifies a subset of columns of CSV data to select. If
      column names are provided, these must correspond to names provided in
      `column_names` or inferred from the file header lines. When this argument
      is specified, only a subset of CSV columns will be parsed and returned,
      corresponding to the columns specified. Using this results in faster
      parsing and lower memory usage. If both this and `column_defaults` are
      specified, these must have the same lengths, and `column_defaults` is
      assumed to be sorted in order of increasing column index.
    field_delim: An optional `string`. Defaults to `","`. Char delimiter to
      separate fields in a record.
    use_quote_delim: An optional bool. Defaults to `True`. If false, treats
      double quotation marks as regular characters inside of the string fields.
    na_value: Additional string to recognize as NA/NaN.
    header: A bool that indicates whether the first rows of provided CSV files
      correspond to header lines with column names, and should not be included
      in the data.
    num_epochs: An int specifying the number of times this dataset is repeated.
      If None, cycles through the dataset forever.
    shuffle: A bool that indicates whether the input should be shuffled.
    shuffle_buffer_size: Buffer size to use for shuffling. A large buffer size
      ensures better shuffling, but increases memory usage and startup time.
    shuffle_seed: Randomization seed to use for shuffling.
    prefetch_buffer_size: An int specifying the number of feature
      batches to prefetch for performance improvement. Recommended value is the
      number of batches consumed per training step. Defaults to auto-tune.
    num_parallel_reads: Number of threads used to read CSV records from files.
      If >1, the results will be interleaved. Defaults to `1`.
    sloppy: If `True`, reading performance will be improved at
      the cost of non-deterministic ordering. If `False`, the order of elements
      produced is deterministic prior to shuffling (elements are still
      randomized if `shuffle=True`. Note that if the seed is set, then order
      of elements after shuffling is deterministic). Defaults to `False`.
    num_rows_for_inference: Number of rows of a file to use for type inference
      if record_defaults is not provided. If None, reads all the rows of all
      the files. Defaults to 100.
    compression_type: (Optional.) A `tf.string` scalar evaluating to one of
      `""` (no compression), `"ZLIB"`, or `"GZIP"`. Defaults to no compression.
    ignore_errors: (Optional.) If `True`, ignores errors with CSV file parsing,
      such as malformed data or empty lines, and moves on to the next valid
      CSV record. Otherwise, the dataset raises an error and stops processing
      when encountering any invalid records. Defaults to `False`.
    encoding: Encoding to use when reading. Defaults to `UTF-8`.

  Returns:
    A dataset, where each element is a (features, labels) tuple that corresponds
    to a batch of `batch_size` CSV rows. The features dictionary maps feature
    column names to `Tensor`s containing the corresponding column data, and
    labels is a `Tensor` containing the column data for the label column
    specified by `label_name`.

  Raises:
    ValueError: If any of the arguments is malformed.
  """
    if num_parallel_reads is None:
        num_parallel_reads = 1
    if prefetch_buffer_size is None:
        prefetch_buffer_size = dataset_ops.AUTOTUNE
    filenames = _get_file_names(file_pattern, False)
    dataset = dataset_ops.Dataset.from_tensor_slices(filenames)
    if shuffle:
        dataset = dataset.shuffle(len(filenames), shuffle_seed)
    if column_names is None or column_defaults is None:
        file_io_fn = lambda filename: file_io.FileIO(filename, 'r', encoding=encoding)
        if compression_type is not None:
            compression_type_value = tensor_util.constant_value(compression_type)
            if compression_type_value is None:
                raise ValueError(f'Received unknown `compression_type` {compression_type}. Expected: GZIP, ZLIB or  (empty string).')
            if compression_type_value == 'GZIP':
                file_io_fn = lambda filename: gzip.open(filename, 'rt', encoding=encoding)
            elif compression_type_value == 'ZLIB':
                raise ValueError(f'`compression_type` {compression_type} is not supported for probing columns.')
            elif compression_type_value != '':
                raise ValueError(f'Received unknown `compression_type` {compression_type}. Expected: GZIP, ZLIB or  (empty string).')
    if column_names is None:
        if not header:
            raise ValueError('Expected `column_names` or `header` arguments. Neither is provided.')
        column_names = _infer_column_names(filenames, field_delim, use_quote_delim, file_io_fn)
    if len(column_names) != len(set(column_names)):
        sorted_names = sorted(column_names)
        duplicate_columns = set([a for a, b in zip(sorted_names[:-1], sorted_names[1:]) if a == b])
        raise ValueError(f'Either `column_names` argument or CSV header row contains duplicate column names: {duplicate_columns}.')
    if select_columns is not None:
        select_columns = _get_sorted_col_indices(select_columns, column_names)
    if column_defaults is not None:
        column_defaults = [constant_op.constant([], dtype=x) if not tensor_util.is_tf_type(x) and x in _ACCEPTABLE_CSV_TYPES else x for x in column_defaults]
    else:
        column_defaults = _infer_column_defaults(filenames, len(column_names), field_delim, use_quote_delim, na_value, header, num_rows_for_inference, select_columns, file_io_fn)
    if select_columns is not None and len(column_defaults) != len(select_columns):
        raise ValueError(f'If specified, `column_defaults` and `select_columns` must have the same length: `column_defaults` has length {len(column_defaults)}, `select_columns` has length {len(select_columns)}.')
    if select_columns is not None and len(column_names) > len(select_columns):
        column_names = [column_names[i] for i in select_columns]
    if label_name is not None and label_name not in column_names:
        raise ValueError(f'`label_name` provided must be one of the columns: {column_names}. Received: {label_name}.')

    def filename_to_dataset(filename):
        dataset = CsvDataset(filename, record_defaults=column_defaults, field_delim=field_delim, use_quote_delim=use_quote_delim, na_value=na_value, select_cols=select_columns, header=header, compression_type=compression_type)
        if ignore_errors:
            dataset = dataset.apply(error_ops.ignore_errors())
        return dataset

    def map_fn(*columns):
        """Organizes columns into a features dictionary.

    Args:
      *columns: list of `Tensor`s corresponding to one csv record.
    Returns:
      An OrderedDict of feature names to values for that particular record. If
      label_name is provided, extracts the label feature to be returned as the
      second element of the tuple.
    """
        features = collections.OrderedDict(zip(column_names, columns))
        if label_name is not None:
            label = features.pop(label_name)
            return (features, label)
        return features
    if num_parallel_reads == dataset_ops.AUTOTUNE:
        dataset = dataset.interleave(filename_to_dataset, num_parallel_calls=num_parallel_reads)
        options = options_lib.Options()
        options.deterministic = not sloppy
        dataset = dataset.with_options(options)
    else:

        def apply_fn(dataset):
            return core_readers.ParallelInterleaveDataset(dataset, filename_to_dataset, cycle_length=num_parallel_reads, block_length=1, sloppy=sloppy, buffer_output_elements=None, prefetch_input_elements=None)
        dataset = dataset.apply(apply_fn)
    dataset = _maybe_shuffle_and_repeat(dataset, num_epochs, shuffle, shuffle_buffer_size, shuffle_seed)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=num_epochs is None)
    dataset = map_op._MapDataset(dataset, map_fn, use_inter_op_parallelism=False)
    dataset = dataset.prefetch(prefetch_buffer_size)
    return dataset