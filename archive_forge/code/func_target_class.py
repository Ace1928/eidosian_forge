import weakref
from tensorflow.python.util import tf_inspect
@property
def target_class(self):
    true_self = self.weakrefself_target__()
    if tf_inspect.isclass(true_self):
        return true_self
    else:
        return true_self.__class__