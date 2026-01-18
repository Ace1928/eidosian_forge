from collections import OrderedDict
def resolve_ellipsis(names, tensor_names, fn_name):
    """
    Expands ... inside `names` to be equal to a list of names from `tensor_names`.
    """
    ellipsis_idx = single_ellipsis_index(names, fn_name)
    if ellipsis_idx is None:
        return names
    return replace_ellipsis_by_position(ellipsis_idx, names, tensor_names)