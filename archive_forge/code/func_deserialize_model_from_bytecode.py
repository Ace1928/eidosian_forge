import os
import tempfile
import tensorflow.compat.v2 as tf
from keras.src.saving import saving_lib
def deserialize_model_from_bytecode(serialized_model):
    """Reconstruct a Model from the output of `serialize_model_as_bytecode`.

    Args:
        serialized_model: (bytes) return value from
          `serialize_model_as_bytecode`.

    Returns:
        Keras Model instance.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        filepath = os.path.join(temp_dir, 'model.keras')
        with open(filepath, 'wb') as f:
            f.write(serialized_model)
        model = saving_lib.load_model(filepath, safe_mode=False)
    except Exception as e:
        raise e
    else:
        return model
    finally:
        tf.io.gfile.rmtree(temp_dir)