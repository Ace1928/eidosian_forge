import threading
import time
from collections import OrderedDict
from threading import RLock
from time import sleep
from typing import Any, Tuple
from adagio.exceptions import AbortedError
from adagio.instances import (NoOpCache, ParallelExecutionEngine, TaskContext,
from adagio.shells.interfaceless import function_to_taskspec
from adagio.specs import InputSpec, OutputSpec, WorkflowSpec, _NodeSpec
from pytest import raises
from triad.exceptions import InvalidOperationError
from timeit import timeit
def t4(ctx: TaskContext):
    a = ctx.inputs.get_or_throw('a', int)
    b = ctx.configs.get_or_throw('b', str)
    ctx.outputs['_0'] = a + len(b)
    for i in range(5):
        time.sleep(0.1)
        if ctx.abort_requested:
            raise AbortedError()
    raise SyntaxError('Expected')