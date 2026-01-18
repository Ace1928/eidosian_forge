import collections
import enum
import functools
from typing import Text
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.lite.python.metrics import metrics
def report_error_message(error_message: Text):
    error_data = converter_error_data_pb2.ConverterErrorData()
    error_data.error_message = error_message
    report_error(error_data)