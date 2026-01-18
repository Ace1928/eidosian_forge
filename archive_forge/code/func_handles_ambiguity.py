from typing import Type, AbstractSet
from random import randint
from collections import deque
from operator import attrgetter
from importlib import import_module
from functools import partial
from ..parse_tree_builder import AmbiguousIntermediateExpander
from ..visitors import Discard
from ..utils import logger, OrderedSet
from ..tree import Tree
def handles_ambiguity(func):
    """Decorator for methods of subclasses of ``TreeForestTransformer``.
    Denotes that the method should receive a list of transformed derivations."""
    func.handles_ambiguity = True
    return func