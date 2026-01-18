import importlib.metadata
import warnings
from copy import deepcopy
from packaging import version
from ..utils import is_accelerate_available, is_bitsandbytes_available, logging
def get_keys_to_not_convert(model):
    """
    An utility function to get the key of the module to keep in full precision if any For example for CausalLM modules
    we may want to keep the lm_head in full precision for numerical stability reasons. For other architectures, we want
    to keep the tied weights of the model. The function will return a list of the keys of the modules to not convert in
    int8.

    Parameters:
    model (`torch.nn.Module`):
        Input model
    """
    tied_model = deepcopy(model)
    tied_model.tie_weights()
    tied_params = find_tied_parameters(tied_model)
    if isinstance(tied_params, dict):
        tied_keys = sum(list(tied_params.values()), []) + list(tied_params.keys())
    else:
        tied_keys = sum(tied_params, [])
    has_tied_params = len(tied_keys) > 0
    if not has_tied_params:
        output_emb = model.get_output_embeddings()
        if output_emb is not None:
            list_last_module = [name for name, module in model.named_modules() if id(module) == id(output_emb)]
            return list_last_module
    list_modules = list(model.named_parameters())
    list_last_module = [list_modules[-1][0]]
    intersection = set(list_last_module) - set(tied_keys)
    list_untouched = list(set(tied_keys)) + list(intersection)
    names_to_remove = ['.weight', '.bias']
    filtered_module_names = []
    for name in list_untouched:
        for name_to_remove in names_to_remove:
            if name_to_remove in name:
                name = name.replace(name_to_remove, '')
        filtered_module_names.append(name)
    return filtered_module_names