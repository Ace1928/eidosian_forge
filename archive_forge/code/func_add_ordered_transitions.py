from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def add_ordered_transitions(self, states=None, trigger='next_state', loop=True, loop_includes_initial=True, conditions=None, unless=None, before=None, after=None, prepare=None, **kwargs):
    if states is None:
        states = self.get_nested_state_names()
    super(HierarchicalMachine, self).add_ordered_transitions(states=states, trigger=trigger, loop=loop, loop_includes_initial=loop_includes_initial, conditions=conditions, unless=unless, before=before, after=after, prepare=prepare, **kwargs)