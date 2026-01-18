import threading
from tensorboard import errors
def parse_tensor_name(tensor_name):
    """Helper function that extracts op name and slot from tensor name."""
    output_slot = 0
    if ':' in tensor_name:
        op_name, output_slot = tensor_name.split(':')
        output_slot = int(output_slot)
    else:
        op_name = tensor_name
    return (op_name, output_slot)