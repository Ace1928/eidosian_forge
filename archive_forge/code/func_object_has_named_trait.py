from traits.constants import ComparisonMode, TraitKind
from traits.ctraits import CHasTraits
from traits.observation._observe import add_or_remove_notifiers
from traits.observation.exceptions import NotifierNotFound
from traits.trait_base import Undefined, Uninitialized
def object_has_named_trait(object, name):
    """ Return true if a trait with the given name is defined on the object.

    Parameters
    ----------
    object : any
        Any object
    name : str
        Trait name to look for.

    Returns
    -------
    boolean
    """
    return isinstance(object, CHasTraits) and object._trait(name, 0) is not None