import inspect
from . import ctraits
from .constants import ComparisonMode, DefaultValue, default_value_map
from .observation.i_observable import IObservable
from .trait_base import SequenceTypes, Undefined
from .trait_dict_object import TraitDictObject
from .trait_list_object import TraitListObject
from .trait_set_object import TraitSetObject
@property_fields.setter
def property_fields(self, value):
    """ Set the fget, fset, validate callables for the property.

        Parameters
        ----------
        value : tuple
            Value should be the tuple of callables (fget, fset, validate).

        """
    func_arg_counts = []
    for arg in value:
        if arg is None:
            nargs = 0
        else:
            sig = inspect.signature(arg)
            nargs = len(sig.parameters)
        func_arg_counts.extend([arg, nargs])
    self._set_property(*func_arg_counts)