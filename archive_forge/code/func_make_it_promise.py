import inspect
from functools import partial
from itertools import chain
from wandb_promise import Promise
def make_it_promise(next, *a, **b):
    return Promise.resolve(next(*a, **b))