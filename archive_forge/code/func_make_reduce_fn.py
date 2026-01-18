import collections
import copy
import functools
import itertools
import operator
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List
import torch
from transformers.file_utils import add_end_docstrings
from transformers.utils.fx import _gen_constructor_wrapper
def make_reduce_fn(reverse):

    def reduce_fn(f, g):

        def composition(graph_module, lint_and_recompile=False, reverse=reverse):
            return f(g(graph_module, lint_and_recompile=lint_and_recompile, reverse=reverse), lint_and_recompile=lint_and_recompile, reverse=reverse)
        return composition
    return reduce_fn