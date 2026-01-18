import sys
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
@_converter_requires('torch')
def convert_metric_value_to_float_if_torch_tensor(x):
    import torch
    if isinstance(x, torch.Tensor):
        extracted_tensor_val = x.detach().cpu()
        return float(_try_get_item(extracted_tensor_val))
    return x