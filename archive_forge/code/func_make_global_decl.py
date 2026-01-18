from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union
from dataclasses import dataclass
def make_global_decl(meta: KernelLinkerMeta) -> str:
    return f'\nCUresult {meta.orig_kernel_name}_default(CUstream stream, {gen_signature_with_full_args(meta)});\nCUresult {meta.orig_kernel_name}(CUstream stream, {gen_signature_with_full_args(meta)}, int algo_id);\nvoid load_{meta.orig_kernel_name}();\nvoid unload_{meta.orig_kernel_name}();\n    '