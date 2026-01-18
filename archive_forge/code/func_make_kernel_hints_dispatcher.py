from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union
from dataclasses import dataclass
def make_kernel_hints_dispatcher(name: str, metas: Sequence[KernelLinkerMeta]) -> str:
    src = f'// launcher for: {name}\n'
    for meta in sorted(metas, key=lambda m: -m.num_specs):
        src += f'CUresult {meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}(CUstream stream, {gen_signature(meta)});\n'
    src += '\n'
    src += f'CUresult {name}(CUstream stream, {gen_signature_with_full_args(metas[-1])}){{'
    src += '\n'
    for meta in sorted(metas, key=lambda m: -m.num_specs):
        cond_fn = lambda val, hint: f'({val} % {hint} == 0)' if hint == 16 else f'({val} == {hint})' if hint == 1 else None
        conds = ' && '.join([cond_fn(val, hint) for val, hint in zip(meta.arg_names, meta.sizes) if hint is not None])
        src += f'  if ({conds})\n' if any(meta.sizes) else 'if (1)\n'
        arg_names = [arg for arg, hint in zip(meta.arg_names, meta.sizes) if hint != 1]
        src += f'    return {meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}(stream, {', '.join(arg_names)});\n'
    src += '\n'
    src += '  return CUDA_ERROR_INVALID_VALUE;\n'
    src += '}\n'
    for mode in ['load', 'unload']:
        src += f'\n// {mode} for: {name}\n'
        for meta in sorted(metas, key=lambda m: -m.num_specs):
            src += f'void {mode}_{meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}();\n'
        src += f'void {mode}_{name}() {{'
        src += '\n'
        for meta in sorted(metas, key=lambda m: -m.num_specs):
            src += f'  {mode}_{meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}();\n'
        src += '}\n'
    return src