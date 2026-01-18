from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union
from dataclasses import dataclass
def make_func_pointers(names: str, meta: KernelLinkerMeta) -> str:
    src = f'typedef CUresult (*kernel_func_t)(CUstream stream, {gen_signature_with_full_args(meta)});\n'
    src += f'kernel_func_t {meta.orig_kernel_name}_kernels[] = {{\n'
    for name in names:
        src += f'  {name},\n'
    src += '};\n'
    return src