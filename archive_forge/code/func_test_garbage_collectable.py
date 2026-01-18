import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
def test_garbage_collectable(self):
    instance = ClassWithPropertyObservesDefault()
    instance_ref = weakref.ref(instance)
    del instance
    self.assertIsNone(instance_ref())