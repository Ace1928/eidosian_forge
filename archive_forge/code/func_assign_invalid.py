import unittest
from traits.api import HasTraits, Subclass, TraitError, Type
def assign_invalid():
    example_model._class = UnrelatedClass