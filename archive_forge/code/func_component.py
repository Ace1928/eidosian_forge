import collections
import enum
import functools
from typing import Text
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.lite.python.metrics import metrics
@property
def component(self):
    return self.value.component