import torch._C._lazy
def render_ir_graph(tensors):
    """Return a text dump of the LTC IR graph in dot format for the tensors.
    The text can be processed by tools like dot to be rendered in pdf,png etc."""
    return torch._C._lazy._get_tensors_dot(tensors)