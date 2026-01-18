import sys
from keras.src import backend as backend_module
from keras.src.backend.common import global_state
def set_backend(self, backend):
    self._backend = backend