from tensorflow.python.saved_model.registration.registration import get_registered_class
from tensorflow.python.saved_model.registration.registration import get_registered_class_name
from tensorflow.python.saved_model.registration.registration import get_registered_saver_name
from tensorflow.python.saved_model.registration.registration import get_restore_function
from tensorflow.python.saved_model.registration.registration import get_save_function
from tensorflow.python.saved_model.registration.registration import get_strict_predicate_restore
from tensorflow.python.saved_model.registration.registration import register_checkpoint_saver
from tensorflow.python.saved_model.registration.registration import register_serializable
from tensorflow.python.saved_model.registration.registration import RegisteredSaver
from tensorflow.python.saved_model.registration.registration import validate_restore_function
def register_tf_serializable(name=None, predicate=None):
    """See the docstring for `register_serializable`."""
    return register_serializable(package='tf', name=name, predicate=predicate)