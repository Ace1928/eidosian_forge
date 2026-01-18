from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def get_nested_transitions(self, trigger='', src_path=None, dest_path=None):
    """ Retrieves a list of all transitions matching the passed requirements.
        Args:
            trigger (str): If set, return only transitions related to this trigger.
            src_path (list(str)): If set, return only transitions with this source state.
            dest_path (list(str)): If set, return only transitions with this destination.

        Returns:
            list(NestedTransitions) of valid transitions.
        """
    if src_path and dest_path:
        src = self.state_cls.separator.join(src_path)
        dest = self.state_cls.separator.join(dest_path)
        transitions = super(HierarchicalMachine, self).get_transitions(trigger, src, dest)
        if len(src_path) > 1 and len(dest_path) > 1:
            with self(src_path[0]):
                transitions.extend(self.get_nested_transitions(trigger, src_path[1:], dest_path[1:]))
    elif src_path:
        src = self.state_cls.separator.join(src_path)
        transitions = super(HierarchicalMachine, self).get_transitions(trigger, src, '*')
        if len(src_path) > 1:
            with self(src_path[0]):
                transitions.extend(self.get_nested_transitions(trigger, src_path[1:], None))
    elif dest_path:
        dest = self.state_cls.separator.join(dest_path)
        transitions = super(HierarchicalMachine, self).get_transitions(trigger, '*', dest)
        if len(dest_path) > 1:
            for state_name in self.states:
                with self(state_name):
                    transitions.extend(self.get_nested_transitions(trigger, None, dest_path[1:]))
    else:
        transitions = super(HierarchicalMachine, self).get_transitions(trigger, '*', '*')
        for state_name in self.states:
            with self(state_name):
                transitions.extend(self.get_nested_transitions(trigger, None, None))
    return transitions