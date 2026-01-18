import logging
from functools import partial
from transitions import Transition
from ..core import listify
from .markup import MarkupMachine, HierarchicalMarkupMachine
from .nesting import NestedTransition
def add_transition(self, trigger, source, dest, conditions=None, unless=None, before=None, after=None, prepare=None, **kwargs):
    """ Calls the base method and regenerates all models's graphs. """
    super(GraphMachine, self).add_transition(trigger, source, dest, conditions=conditions, unless=unless, before=before, after=after, prepare=prepare, **kwargs)
    for model in self.models:
        model.get_graph(force_new=True)