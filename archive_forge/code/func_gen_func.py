import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
def gen_func(self, n_args, n_kws):
    """
            Generates a function that calls sum_jit_func
            with the desired number of args and kws.
        """
    param_list = [f'arg{i}' for i in range(n_args + n_kws)]
    args_list = []
    for i in range(n_args + n_kws):
        if i % 5 == 0:
            arg_val = f'pow(arg{i}, 2)'
        else:
            arg_val = f'arg{i}'
        args_list.append(arg_val)
    total_params = ', '.join(param_list)
    func_text = f'def impl({total_params}):\n'
    func_text += '    return sum_jit_func(\n'
    for i in range(n_args):
        func_text += f'        {args_list[i]},\n'
    for i in range(n_args, n_args + n_kws):
        func_text += f'        {param_list[i]}={args_list[i]},\n'
    func_text += '    )\n'
    local_vars = {}
    exec(func_text, {'sum_jit_func': sum_jit_func}, local_vars)
    return local_vars['impl']