import os
import tempfile
import shutil
from io import StringIO
from sympy.core import symbols, Eq
from sympy.utilities.autowrap import (autowrap, binary_function,
from sympy.utilities.codegen import (
from sympy.testing.pytest import raises
from sympy.testing.tmpfiles import TmpFileManager
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import numpy as np
def test_cython_wrapper_compile_flags():
    from sympy.core.relational import Equality
    x, y, z = symbols('x,y,z')
    routine = make_routine('test', Equality(z, x + y))
    code_gen = CythonCodeWrapper(CCodeGen())
    expected = "from setuptools import setup\nfrom setuptools import Extension\nfrom Cython.Build import cythonize\ncy_opts = {'compiler_directives': {'language_level': '3'}}\n\next_mods = [Extension(\n    'wrapper_module_%(num)s', ['wrapper_module_%(num)s.pyx', 'wrapped_code_%(num)s.c'],\n    include_dirs=[],\n    library_dirs=[],\n    libraries=[],\n    extra_compile_args=['-std=c99'],\n    extra_link_args=[]\n)]\nsetup(ext_modules=cythonize(ext_mods, **cy_opts))\n" % {'num': CodeWrapper._module_counter}
    temp_dir = tempfile.mkdtemp()
    TmpFileManager.tmp_folder(temp_dir)
    setup_file_path = os.path.join(temp_dir, 'setup.py')
    code_gen._prepare_files(routine, build_dir=temp_dir)
    with open(setup_file_path) as f:
        setup_text = f.read()
    assert setup_text == expected
    code_gen = CythonCodeWrapper(CCodeGen(), include_dirs=['/usr/local/include', '/opt/booger/include'], library_dirs=['/user/local/lib'], libraries=['thelib', 'nilib'], extra_compile_args=['-slow-math'], extra_link_args=['-lswamp', '-ltrident'], cythonize_options={'compiler_directives': {'boundscheck': False}})
    expected = "from setuptools import setup\nfrom setuptools import Extension\nfrom Cython.Build import cythonize\ncy_opts = {'compiler_directives': {'boundscheck': False}}\n\next_mods = [Extension(\n    'wrapper_module_%(num)s', ['wrapper_module_%(num)s.pyx', 'wrapped_code_%(num)s.c'],\n    include_dirs=['/usr/local/include', '/opt/booger/include'],\n    library_dirs=['/user/local/lib'],\n    libraries=['thelib', 'nilib'],\n    extra_compile_args=['-slow-math', '-std=c99'],\n    extra_link_args=['-lswamp', '-ltrident']\n)]\nsetup(ext_modules=cythonize(ext_mods, **cy_opts))\n" % {'num': CodeWrapper._module_counter}
    code_gen._prepare_files(routine, build_dir=temp_dir)
    with open(setup_file_path) as f:
        setup_text = f.read()
    assert setup_text == expected
    expected = "from setuptools import setup\nfrom setuptools import Extension\nfrom Cython.Build import cythonize\ncy_opts = {'compiler_directives': {'boundscheck': False}}\nimport numpy as np\n\next_mods = [Extension(\n    'wrapper_module_%(num)s', ['wrapper_module_%(num)s.pyx', 'wrapped_code_%(num)s.c'],\n    include_dirs=['/usr/local/include', '/opt/booger/include', np.get_include()],\n    library_dirs=['/user/local/lib'],\n    libraries=['thelib', 'nilib'],\n    extra_compile_args=['-slow-math', '-std=c99'],\n    extra_link_args=['-lswamp', '-ltrident']\n)]\nsetup(ext_modules=cythonize(ext_mods, **cy_opts))\n" % {'num': CodeWrapper._module_counter}
    code_gen._need_numpy = True
    code_gen._prepare_files(routine, build_dir=temp_dir)
    with open(setup_file_path) as f:
        setup_text = f.read()
    assert setup_text == expected
    TmpFileManager.cleanup()