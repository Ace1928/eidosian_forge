from functools import partial
from .constants import DefaultValue
def mapped_trait_for(trait, name):
    """ Returns the 'mapped trait' definition for a mapped trait, the default
        value of which is a callable that maps the value of the original trait.

        Parameters
        ----------
        trait : ctrait.CTrait
            A trait for which the 'mapped trait' definition is being created.
        name : str
            The name of the trait for which the 'mapped trait' definition is
            being created.

        Returns
        -------
        trait_types.Any
            A definition of the 'mapped trait'
    """
    from .trait_types import Any
    mapped_trait = Any(is_base=False, transient=True, editable=False).as_ctrait()
    mapped_trait.set_default_value(DefaultValue.callable, partial(_mapped_trait_default, trait, name))
    return mapped_trait