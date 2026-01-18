import os
import tempfile
import tensorflow.compat.v2 as tf
from keras.src.saving import saving_lib
def serialize_model_as_bytecode(model):
    """Convert a Keras Model into a bytecode representation for pickling.

    Args:
        model: Keras Model instance.

    Returns:
        Tuple that can be read by `deserialize_from_bytecode`.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        filepath = os.path.join(temp_dir, 'model.keras')
        saving_lib.save_model(model, filepath)
        with open(filepath, 'rb') as f:
            data = f.read()
    except Exception as e:
        raise e
    else:
        return data
    finally:
        tf.io.gfile.rmtree(temp_dir)