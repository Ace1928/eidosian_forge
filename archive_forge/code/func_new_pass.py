from functools import wraps
from inspect import unwrap
from typing import Callable, List, Optional
import logging
@wraps(base_pass)
def new_pass(source):
    output = source
    if n_iter is not None and n_iter > 0:
        for _ in range(n_iter):
            output = base_pass(output)
    elif predicate is not None:
        while predicate(output):
            output = base_pass(output)
    else:
        raise RuntimeError(f'loop_pass must be given positive int n_iter (given {n_iter}) xor predicate (given {predicate})')
    return output