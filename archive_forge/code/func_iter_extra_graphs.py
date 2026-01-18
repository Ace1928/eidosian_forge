from traits.observation._trait_change_event import trait_event_factory
from traits.observation._has_traits_helpers import (
from traits.observation._i_observer import IObserver
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation._trait_added_observer import TraitAddedObserver
from traits.observation._trait_event_notifier import TraitEventNotifier
def iter_extra_graphs(self, graph):
    """ Yield additional ObserverGraph for adding/removing notifiers when
        this observer is encountered in a given ObserverGraph.

        Parameters
        ----------
        graph : ObserverGraph
            The graph where this observer is the root node.

        Yields
        ------
        graph : ObserverGraph
        """
    yield ObserverGraph(node=TraitAddedObserver(match_func=lambda name, trait: name == self.name, optional=self.optional), children=[graph])