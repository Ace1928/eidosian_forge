from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union
from dataclasses import dataclass
def make_algo_decls(name: str, metas: Sequence[KernelLinkerMeta]) -> str:
    return f'\nCUresult {name}(CUstream stream, {gen_signature_with_full_args(metas[-1])});\nvoid load_{name}();\nvoid unload_{name}();\n    '