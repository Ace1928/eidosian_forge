import os
import pathlib
import torch
from torch.jit._recursive import wrap_cpp_module
from torch.serialization import validate_cuda_device
def get_flatbuffer_module_info(path_or_file):
    """Get some information regarding a model file in flatbuffer format.

    Args:
        path_or_file: Either str, Path or file like object (BytesIO OK).
            If it's str or Path, we will read the file referenced by that
            path as Bytes.

    Returns:
        A dict with metadata on what that file contains, currently looks like
        this:
        {
            'bytecode_version': 4,  # int
            'operator_version': 4,  # int
            'function_names': {
                '__torch__.___torch_mangle_0.Foo.forward'}, # set
            'type_names': set(),  # set
            'opname_to_num_args': {'aten::linear': 3} # Dict[str, int]
        }
    """
    if isinstance(path_or_file, (str, pathlib.Path)):
        with open(path_or_file, 'rb') as f:
            all_bytes = f.read()
    else:
        all_bytes = path_or_file.read()
    return torch._C._get_module_info_from_flatbuffer(all_bytes)