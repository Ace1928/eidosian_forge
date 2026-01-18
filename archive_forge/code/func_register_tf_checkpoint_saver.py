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
def register_tf_checkpoint_saver(name=None, predicate=None, save_fn=None, restore_fn=None, strict_predicate_restore=True):
    """See the docstring for `register_checkpoint_saver`."""
    return register_checkpoint_saver(package='tf', name=name, predicate=predicate, save_fn=save_fn, restore_fn=restore_fn, strict_predicate_restore=strict_predicate_restore)