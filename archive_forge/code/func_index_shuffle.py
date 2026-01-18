import functools
import numpy as np
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import random_seed
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def index_shuffle(file_infos, reader_factory, seed=None, reshuffle_each_iteration=False, num_parallel_calls=dataset_ops.AUTOTUNE):
    """Creates a (globally) shuffled dataset from the given set of files.

  Unlike `tf.data.Dataset.shuffle()`, which uses an in-memory buffer to shuffle
  elements of input dataset in a streaming fashion,
  `tf.data.experimental.index_shuffle()` performs a global shuffle of element
  indices and then reads the data in a shuffled order. The advantage of
  `index_shuffle()` is that it can perform global shuffle of datasets that do
  not fit into memory (as long as the array of their indices does) and that the
  shuffling logic it provides is compatible with symbolic checkpointing. The
  disadvantage of `index_shuffle()` is that reading data in a shuffled random
  order will in general not be as efficient as reading data sequentially.

  Args:
    file_infos: A list of dictionaries that describe each file of the input
      dataset. Each dictionary is expected to contain the "path" key, which
      identifies the path of the file and the "num_elements" key, which
      identifies the number of elements in the file. In addition, the "skip"
      and "take" keys can be used to identify the number of elements to skip
      and take respectively. By default, no elements are skipped and all
      elements are taken.
    reader_factory: A function that maps a sequence of filenames to an instance
      of `tf.data.Dataset` that reads data from the files.
    seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
      seed that will be used to shuffle the order of elements. Default to
      non-deterministic seed.
    reshuffle_each_iteration: (Optional.) A `tf.bool` scalar `tf.Tensor`, that
      determines whether to change the shuffle order each iteration. Defaults to
      `False`.
    num_parallel_calls: (Optional.) A `tf.int64` scalar `tf.Tensor`, that
      determines the maximum number of random access operations to perform
      in parallel. By default, the tf.data runtime uses autotuning to determine
      the value dynamically.

  Returns:
    A `tf.data.Dataset` object, representing a globally shuffled dataset of
    the input data.
  """
    result = _process_file_infos(file_infos)

    def sequential_index_shuffle(seeds):
        dataset = dataset_ops.Dataset.range(result['num_elements'])

        def read_element(dataset, index):
            shuffled_index = stateless_random_ops.index_shuffle(index, seeds, result['num_elements'] - 1)
            if 'thresholds' in result and 'offsets' in result:
                shuffled_index = _adjust_index(shuffled_index, result['thresholds'], result['offsets'])
            return random_access.at(dataset, shuffled_index)
        map_func = functools.partial(read_element, reader_factory(result['files']))
        return dataset.map(map_func, num_parallel_calls=num_parallel_calls)
    rng_ds = dataset_ops.Dataset.random(seed=seed, rerandomize_each_iteration=reshuffle_each_iteration)
    rng_ds = rng_ds.take(2).batch(2, drop_remainder=True)
    return rng_ds.flat_map(sequential_index_shuffle)