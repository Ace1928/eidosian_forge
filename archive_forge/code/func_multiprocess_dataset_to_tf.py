import os
import warnings
from functools import partial
from math import ceil
from uuid import uuid4
import numpy as np
import pyarrow as pa
from multiprocess import get_context
from .. import config
def multiprocess_dataset_to_tf(dataset, cols_to_retain, collate_fn, collate_fn_args, columns_to_np_types, output_signature, shuffle, batch_size, drop_remainder, num_workers):
    """Create a tf.data.Dataset from the underlying Dataset. This is a multi-process method - the single-process
    equivalent is dataset_to_tf.

    Args:
        dataset (`Dataset`): Dataset to wrap with tf.data.Dataset.
        cols_to_retain (`List[str]`): Dataset column(s) to load in the
            tf.data.Dataset. It is acceptable to include column names that are created by the `collate_fn` and
            that do not exist in the original dataset.
        collate_fn(`Callable`): A function or callable object (such as a `DataCollator`) that will collate
            lists of samples into a batch.
        collate_fn_args (`Dict`): A  `dict` of keyword arguments to be passed to the
            `collate_fn`. Can be empty.
        columns_to_np_types (`Dict[str, np.dtype]`): A `dict` mapping column names to numpy dtypes.
        output_signature (`Dict[str, tf.TensorSpec]`): A `dict` mapping column names to
            `tf.TensorSpec` objects.
        shuffle(`bool`): Shuffle the dataset order when loading. Recommended True for training, False for
            validation/evaluation.
        batch_size (`int`, default `None`): Size of batches to load from the dataset. Defaults to `None`, which implies that
            the dataset won't be batched, but the returned dataset can be batched later with `tf_dataset.batch(batch_size)`.
        drop_remainder(`bool`, default `None`): Drop the last incomplete batch when loading. If not provided,
            defaults to the same setting as shuffle.
        num_workers (`int`): Number of workers to use for loading the dataset. Should be >= 1.

    Returns:
        `tf.data.Dataset`
    """
    if config.TF_AVAILABLE:
        import tensorflow as tf
    else:
        raise ImportError('Called a Tensorflow-specific function but Tensorflow is not installed.')
    data_generator = NumpyMultiprocessingGenerator(dataset=dataset, cols_to_retain=cols_to_retain, collate_fn=collate_fn, collate_fn_args=collate_fn_args, columns_to_np_types=columns_to_np_types, output_signature=output_signature, shuffle=shuffle, batch_size=batch_size, drop_remainder=drop_remainder, num_workers=num_workers)
    tf_dataset = tf.data.Dataset.from_generator(data_generator, output_signature=output_signature)
    if drop_remainder:
        dataset_length = int(len(dataset) // batch_size)
    else:
        dataset_length = int(ceil(len(dataset) / batch_size))
    return tf_dataset.apply(tf.data.experimental.assert_cardinality(dataset_length))