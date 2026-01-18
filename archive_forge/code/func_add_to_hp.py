import re
import warnings
import keras_tuner
import tensorflow as tf
from packaging.version import parse
from tensorflow import nest
def add_to_hp(hp, hps, name=None):
    """Add the HyperParameter (self) to the HyperParameters.

    # Arguments
        hp: keras_tuner.HyperParameters.
        name: String. If left unspecified, the hp name is used.
    """
    if not isinstance(hp, keras_tuner.engine.hyperparameters.HyperParameter):
        return hp
    kwargs = hp.get_config()
    if name is None:
        name = hp.name
    kwargs.pop('conditions')
    kwargs.pop('name')
    class_name = hp.__class__.__name__
    func = getattr(hps, class_name)
    return func(name=name, **kwargs)