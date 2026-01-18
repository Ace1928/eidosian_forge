import re
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from keras.src.callbacks import ModelCheckpoint
from keras.src.optimizers import optimizer
from tensorflow.python.util.tf_export import keras_export
def list_checkpoint_attributes(ckpt_dir_or_file):
    """Lists all the attributes in a checkpoint.

    Checkpoint keys are paths in a checkpoint graph, and attribute is the first
    element in the path. e.g. with a checkpoint key
    "optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE", optimizer is the attribute. The
    attribute is also used to save/restore a variable in a checkpoint,
    e.g. tf.train.Checkpoint(optimizer=optimizer, model=model).

    Args:
      ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint.

    Returns:
      Set of attributes in a checkpoint.
    """
    reader = tf.train.load_checkpoint(ckpt_dir_or_file)
    variable_map = reader.get_variable_to_shape_map()
    return {name.split('/')[0] for name in variable_map.keys()}