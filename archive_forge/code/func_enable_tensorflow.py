import warnings
from packaging.version import Version
def enable_tensorflow():
    warn_msg = 'Built-in TensorFlow support will be removed in Thinc v9. If you need TensorFlow support in the future, you can transition to using a custom copy of the current TensorFlowWrapper in your package or project.'
    warnings.warn(warn_msg, DeprecationWarning)
    global tensorflow, has_tensorflow, has_tensorflow_gpu
    import tensorflow
    import tensorflow.experimental.dlpack
    has_tensorflow = True
    has_tensorflow_gpu = len(tensorflow.config.get_visible_devices('GPU')) > 0