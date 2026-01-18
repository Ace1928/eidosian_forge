import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.utils import dataset_utils
from tensorflow.python.util.tf_export import keras_export
def paths_and_labels_to_dataset(file_paths, labels, label_mode, num_classes, max_length):
    """Constructs a dataset of text strings and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    string_ds = path_ds.map(lambda x: path_to_string_content(x, max_length), num_parallel_calls=tf.data.AUTOTUNE)
    if label_mode:
        label_ds = dataset_utils.labels_to_dataset(labels, label_mode, num_classes)
        string_ds = tf.data.Dataset.zip((string_ds, label_ds))
    return string_ds