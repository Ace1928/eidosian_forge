import shutil
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.utilities._compilation.compilation import compile_link_import_strings
import numpy as np
def test_compile_link_import_strings():
    if not numpy:
        skip('numpy not installed.')
    if not cython:
        skip('cython not installed.')
    from sympy.utilities._compilation import has_c
    if not has_c():
        skip('No C compiler found.')
    compile_kw = {'std': 'c99', 'include_dirs': [numpy.get_include()]}
    info = None
    try:
        mod, info = compile_link_import_strings(_sources1, compile_kwargs=compile_kw)
        data = numpy.random.random(1024 * 1024 * 8)
        res_mod = mod.sigmoid(data)
        res_npy = npy(data)
        assert numpy.allclose(res_mod, res_npy)
    finally:
        if info and info['build_dir']:
            shutil.rmtree(info['build_dir'])