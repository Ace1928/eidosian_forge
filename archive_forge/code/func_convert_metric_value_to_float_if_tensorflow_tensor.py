import sys
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
@_converter_requires('tensorflow')
def convert_metric_value_to_float_if_tensorflow_tensor(x):
    import tensorflow as tf
    if isinstance(x, tf.Tensor):
        try:
            return float(x)
        except Exception as e:
            raise MlflowException(f'Failed to convert metric value to float: {e!r}', error_code=INVALID_PARAMETER_VALUE)
    return x