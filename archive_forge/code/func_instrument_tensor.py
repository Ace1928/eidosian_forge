import collections
import hashlib
import os
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tensor_tracer_pb2
def instrument_tensor(self, tensor, explanation):
    self.instrument(tensor.name, explanation)