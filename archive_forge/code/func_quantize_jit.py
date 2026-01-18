import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module
def quantize_jit(model, qconfig_dict, run_fn, run_args, inplace=False, debug=False):
    """Quantize the input float TorchScript model with
    post training static quantization.

    First it will prepare the model for calibration, then it calls
    `run_fn` which will run the calibration step, after that we will
    convert the model to a quantized model.

    Args:
        `model`: input float TorchScript model
        `qconfig_dict`: qconfig_dict is a dictionary with names of sub modules as key and
        qconfig for that module as value, empty key means the qconfig will be applied
        to whole model unless it's overwritten by more specific configurations, the
        qconfig for each module is either found in the dictionary or fallback to
         the qconfig of parent module.

        Right now qconfig_dict is the only way to configure how the model is quantized,
        and it is done in the granularity of module, that is, we only support one type
        of qconfig for each torch.nn.Module, and the qconfig for sub module will
        override the qconfig for parent module, empty string means global configuration.
        `run_fn`: a calibration function for calibrating the prepared model
        `run_args`: positional arguments for `run_fn`
        `inplace`: carry out model transformations in-place, the original module is
        mutated
        `debug`: flag for producing a debug friendly model (preserve weight attribute)

    Return:
        Quantized TorchSciprt model.

    Example:
    ```python
    import torch
    from torch.ao.quantization import get_default_qconfig
    from torch.ao.quantization import quantize_jit

    ts_model = torch.jit.script(float_model.eval())  # or torch.jit.trace(float_model, input)
    qconfig = get_default_qconfig('fbgemm')
    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for image, target in data_loader:
                model(image)

    quantized_model = quantize_jit(
        ts_model,
        {'': qconfig},
        calibrate,
        [data_loader_test])
    ```
    """
    torch._C._log_api_usage_once('quantization_api.quantize_jit.quantize_jit')
    return _quantize_jit(model, qconfig_dict, run_fn, run_args, inplace, debug, quant_type=QuantType.STATIC)