from traits.observation._trait_change_event import trait_event_factory
from traits.observation._has_traits_helpers import (
from traits.observation._i_observer import IObserver
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation._trait_added_observer import TraitAddedObserver
from traits.observation._trait_event_notifier import TraitEventNotifier
def iter_objects(self, object):
    """ Yield the value of the named trait from the given object, if the
        value is defined. The value will then be given to the next observer(s)
        following this one in an ObserverGraph.

        If the value has not been set as an instance attribute, i.e. absent in
        ``object.__dict__``, this observer will yield nothing. This is to avoid
        evaluating default initializers while adding observers. When the
        default is defined, a change event will trigger the maintainer to
        add/remove notifiers for the next observers.

        Note that ``Undefined``, ``Uninitialized`` and ``None`` values are also
        skipped, as they are inevitable filled values that contain no further
        attributes to be observed by any other observers.

        Parameters
        ----------
        object: HasTraits
            Expected to be an instance of HasTraits

        Yields
        ------
        value : any

        Raises
        ------
        ValueError
            If the trait is not found and optional flag is set to false.
        """
    if not object_has_named_trait(object, self.name):
        if self.optional:
            return
        raise ValueError('Trait named {!r} not found on {!r}.'.format(self.name, object))
    yield from iter_objects(object, self.name)