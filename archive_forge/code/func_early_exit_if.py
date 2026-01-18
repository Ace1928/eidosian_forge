import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
@contextmanager
def early_exit_if(builder, stack: ExitStack, cond):
    """
    The Python code::

        with contextlib.ExitStack() as stack:
            with early_exit_if(builder, stack, cond):
                cleanup()
            body()

    emits the code::

        if (cond) {
            <cleanup>
        }
        else {
            <body>
        }

    This can be useful for generating code with lots of early exits, without
    having to increase the indentation each time.
    """
    then, otherwise = stack.enter_context(builder.if_else(cond, likely=False))
    with then:
        yield
    stack.enter_context(otherwise)