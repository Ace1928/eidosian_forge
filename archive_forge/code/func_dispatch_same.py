from unittest import mock
from traits.observation._observe import add_or_remove_notifiers
from traits.observation._observer_graph import ObserverGraph
from traits.observation.exceptions import NotifierNotFound
def dispatch_same(handler, event):
    handler(event)