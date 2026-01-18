import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.quantization
import torch.ao.ns._numeric_suite as ns
def bias_correction(float_model, quantized_model, img_data, target_modules=_supported_modules_quantized, neval_batches=None):
    """Perform bias correction on a module.

    Using numeric suite shadow module, the expected output of the floating point and quantized modules
    is recorded. Using that data the bias of supported modules is shifted to compensate for the drift caused
    by quantization
    Paper reference: https://arxiv.org/pdf/1906.04721.pdf (Section 4.2)

    Args:
        float_model: a trained model that serves as a reference to what bias correction should aim for
        quantized_model: quantized form of float_model that bias correction is to applied to
        img_data: calibration data to estimate the expected output (used to find quantization error)
        target_modules: specifies what submodules in quantized_model need bias correction (can be extended to
                unquantized submodules)
        neval_batches: a cap to the number of batches you want to be used for estimating the expected output
    """
    ns.prepare_model_with_stubs(float_model, quantized_model, _supported_modules, MeanShadowLogger)
    uncorrected_modules = {}
    for name, submodule in quantized_model.named_modules():
        if type(submodule) in target_modules:
            uncorrected_modules[name] = submodule
    for uncorrected_module in uncorrected_modules:
        quantized_submodule = get_module(quantized_model, uncorrected_module)
        bias = get_param(quantized_submodule, 'bias')
        if bias is not None:
            count = 0
            for data in img_data:
                quantized_model(data[0])
                count += 1
                if count == neval_batches:
                    break
            ob_dict = ns.get_logger_dict(quantized_model)
            parent_name, _ = parent_child_names(uncorrected_module)
            float_data = ob_dict[parent_name + '.stats']['float']
            quant_data = ob_dict[parent_name + '.stats']['quantized']
            quantization_error = quant_data - float_data
            dims = list(range(quantization_error.dim()))
            dims.remove(1)
            expected_error = torch.mean(quantization_error, dims)
            updated_bias = bias.data - expected_error
            bias.data = updated_bias
            for name, submodule in quantized_model.named_modules():
                if isinstance(submodule, MeanShadowLogger):
                    submodule.clear()