import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def dispatch_here(function, event):
    """ Dispatcher that let the function call through."""
    function(event)