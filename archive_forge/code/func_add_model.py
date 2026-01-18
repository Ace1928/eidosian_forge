import logging
from functools import partial
from transitions import Transition
from ..core import listify
from .markup import MarkupMachine, HierarchicalMarkupMachine
from .nesting import NestedTransition
def add_model(self, model, initial=None):
    models = listify(model)
    super(GraphMachine, self).add_model(models, initial)
    for mod in models:
        mod = self if mod is self.self_literal else mod
        if hasattr(mod, 'get_graph'):
            raise AttributeError('Model already has a get_graph attribute. Graph retrieval cannot be bound.')
        setattr(mod, 'get_graph', partial(self._get_graph, mod))
        _ = mod.get_graph(title=self.title, force_new=True)