import tree
from keras.src.api_export import keras_export
from keras.src.utils.naming import auto_name
def any_symbolic_tensors(args=None, kwargs=None):
    args = args or ()
    kwargs = kwargs or {}
    for x in tree.flatten((args, kwargs)):
        if isinstance(x, KerasTensor):
            return True
    return False