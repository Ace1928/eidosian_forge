from asyncio import iscoroutinefunction
from contextlib import contextmanager
from functools import partial, wraps
from types import coroutine
import builtins
import inspect
import linecache
import logging
import os
import io
import pdb
import subprocess
import sys
import time
import traceback
import warnings
import psutil
@line_cell_magic
def memit(self, line='', cell=None):
    """Measure memory usage of a Python statement

        Usage, in line mode:
          %memit [-r<R>t<T>i<I>] statement

        Usage, in cell mode:
          %%memit [-r<R>t<T>i<I>] setup_code
          code...
          code...

        This function can be used both as a line and cell magic:

        - In line mode you can measure a single-line statement (though multiple
          ones can be chained with using semicolons).

        - In cell mode, the statement in the first line is used as setup code
          (executed but not measured) and the body of the cell is measured.
          The cell body has access to any variables created in the setup code.

        Options:
        -r<R>: repeat the loop iteration <R> times and take the best result.
        Default: 1

        -t<T>: timeout after <T> seconds. Default: None

        -i<I>: Get time information at an interval of I times per second.
            Defaults to 0.1 so that there is ten measurements per second.

        -c: If present, add the memory usage of any children process to the report.

        -o: If present, return a object containing memit run details

        -q: If present, be quiet and do not output a result.

        Examples
        --------
        ::

          In [1]: %memit range(10000)
          peak memory: 21.42 MiB, increment: 0.41 MiB

          In [2]: %memit range(1000000)
          peak memory: 52.10 MiB, increment: 31.08 MiB

          In [3]: %%memit l=range(1000000)
             ...: len(l)
             ...:
          peak memory: 52.14 MiB, increment: 0.08 MiB

        """
    from memory_profiler import memory_usage, _func_exec
    opts, stmt = self.parse_options(line, 'r:t:i:coq', posix=False, strict=False)
    if cell is None:
        setup = 'pass'
    else:
        setup = stmt
        stmt = cell
    repeat = int(getattr(opts, 'r', 1))
    if repeat < 1:
        repeat == 1
    timeout = int(getattr(opts, 't', 0))
    if timeout <= 0:
        timeout = None
    interval = float(getattr(opts, 'i', 0.1))
    include_children = 'c' in opts
    return_result = 'o' in opts
    quiet = 'q' in opts
    import gc
    gc.collect()
    _func_exec(setup, self.shell.user_ns)
    mem_usage = []
    counter = 0
    baseline = memory_usage()[0]
    while counter < repeat:
        counter += 1
        tmp = memory_usage((_func_exec, (stmt, self.shell.user_ns)), timeout=timeout, interval=interval, max_usage=True, max_iterations=1, include_children=include_children)
        mem_usage.append(tmp)
    result = MemitResult(mem_usage, baseline, repeat, timeout, interval, include_children)
    if not quiet:
        if mem_usage:
            print(result)
        else:
            print('ERROR: could not read memory usage, try with a lower interval or more iterations')
    if return_result:
        return result