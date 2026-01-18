import logging
import asyncio
import contextvars
import inspect
from collections import deque
from functools import partial, reduce
import copy
from ..core import State, Condition, Transition, EventData, listify
from ..core import Event, MachineError, Machine
from .nesting import HierarchicalMachine, NestedState, NestedEvent, NestedTransition, resolve_order
def remove_model(self, model):
    """ Remove a model from the state machine. The model will still contain all previously added triggers
        and callbacks, but will not receive updates when states or transitions are added to the Machine.
        If an event queue is used, all queued events of that model will be removed."""
    models = listify(model)
    if self.has_queue == 'model':
        for mod in models:
            del self._transition_queue_dict[id(mod)]
            self.models.remove(mod)
    else:
        for mod in models:
            self.models.remove(mod)
    if len(self._transition_queue) > 0:
        queue = self._transition_queue
        new_queue = [queue.popleft()] + [e for e in queue if e.args[0].model not in models]
        self._transition_queue.clear()
        self._transition_queue.extend(new_queue)