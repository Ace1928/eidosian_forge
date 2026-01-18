import collections
import enum
import functools
from typing import Text
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.lite.python.metrics import metrics
def report_error(error_data: converter_error_data_pb2.ConverterErrorData):
    error_data.component = component.value
    if not error_data.subcomponent:
        error_data.subcomponent = subcomponent.name
    tflite_metrics = metrics.TFLiteConverterMetrics()
    tflite_metrics.set_converter_error(error_data)