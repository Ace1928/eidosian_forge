import inspect
from . import ctraits
from .constants import ComparisonMode, DefaultValue, default_value_map
from .observation.i_observable import IObservable
from .trait_base import SequenceTypes, Undefined
from .trait_dict_object import TraitDictObject
from .trait_list_object import TraitListObject
from .trait_set_object import TraitSetObject
def is_trait_type(self, trait_type):
    """ Returns whether or not this trait is of a specified trait type.
        """
    return isinstance(self.trait_type, trait_type)