import torch
def starcoder_model_postprocess_past_key_value(past_key_values):
    result = []
    for k in past_key_values:
        k = k[:, :, 0]
        k = k.permute([1, 2, 0, 3])
        k = k.reshape(*k.shape[:-2], -1)
        result.append(k)
    return tuple(result)