import logging
from functools import partial
from transitions import Transition
from ..core import listify
from .markup import MarkupMachine, HierarchicalMarkupMachine
from .nesting import NestedTransition
def get_combined_graph(self, title=None, force_new=False, show_roi=False):
    """ This method is currently equivalent to 'get_graph' of the first machine's model.
        In future releases of transitions, this function will return a combined graph with active states
        of all models.
        Args:
            title (str): Title of the resulting graph.
            force_new (bool): Whether a new graph should be generated even if another graph already exists. This should
            be true whenever the model's state or machine's transitions/states/events have changed.
            show_roi (bool): If set to True, only render states that are active and/or can be reached from
                the current state.
        Returns: AGraph (pygraphviz) or Digraph (graphviz) graph instance that can be drawn.
        """
    _LOGGER.info('Returning graph of the first model. In future releases, this method will return a combined graph of all models.')
    return self._get_graph(self.models[0], title, force_new, show_roi)