from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union
from dataclasses import dataclass
def make_default_algo_kernel(meta: KernelLinkerMeta) -> str:
    src = f'CUresult {meta.orig_kernel_name}_default(CUstream stream, {gen_signature_with_full_args(meta)}){{\n'
    src += f'  return {meta.orig_kernel_name}(stream, {', '.join(meta.arg_names)}, 0);\n'
    src += '}\n'
    return src