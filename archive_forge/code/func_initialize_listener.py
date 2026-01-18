import unittest
from unittest import mock
from traits.trait_types import Any, Dict, Event, Str, TraitDictObject
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_errors import TraitError
def initialize_listener(listener):
    """ Initialize a listener so it looks like it hasn't been called.

    This allows us to re-use the listener without having to create and
    wire-up a new one.

    """
    listener.obj = None
    listener.trait_name = None
    listener.old = None
    listener.new = None
    listener.called = 0
    return listener