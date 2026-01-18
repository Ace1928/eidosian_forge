import inspect
from ast import literal_eval
from traitlets import Any, ClassBasedTraitType, TraitError, Undefined
from traitlets.utils.descriptions import describe
def subclass_from_klasses(self, value):
    """Check that a given class is a subclasses found in the klasses list."""
    return any((issubclass(value, klass) for klass in self.importable_klasses))