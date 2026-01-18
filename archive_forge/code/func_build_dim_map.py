from collections import OrderedDict
def build_dim_map(tensor):
    """Returns a map of { dim: dim_name } where dim is a name if the dim is named
    and the dim index otherwise."""
    return OrderedDict([(idx if name is None else name, name) for idx, name in enumerate(tensor.names)])