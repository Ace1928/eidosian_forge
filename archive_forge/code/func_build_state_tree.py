from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def build_state_tree(self, model_states, separator, tree=None):
    """ Converts a list of current states into a hierarchical state tree.
        Args:
            model_states (str or list(str)):
            separator (str): The character used to separate state names
            tree (OrderedDict): The current branch to use. If not passed, create a new tree.
        Returns:
            OrderedDict: A state tree dictionary
        """
    tree = tree if tree is not None else OrderedDict()
    if isinstance(model_states, list):
        for state in model_states:
            _ = self.build_state_tree(state, separator, tree)
    else:
        tmp = tree
        if isinstance(model_states, (Enum, EnumMeta)):
            with self():
                path = self._get_enum_path(model_states)
        else:
            path = model_states.split(separator)
        for elem in path:
            tmp = tmp.setdefault(elem.name if hasattr(elem, 'name') else elem, OrderedDict())
    return tree