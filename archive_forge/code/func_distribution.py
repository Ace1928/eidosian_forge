from tensorflow.python.eager import backprop
from tensorflow.python.training import optimizer as optimizer_lib
@property
def distribution(self):
    return self._distribution