from __future__ import annotations
from typing import (
import warnings
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
def delegate_names(delegate, accessors: list[str], typ: str, overwrite: bool=False, accessor_mapping: Callable[[str], str]=lambda x: x, raise_on_missing: bool=True):
    """
    Add delegated names to a class using a class decorator.  This provides
    an alternative usage to directly calling `_add_delegate_accessors`
    below a class definition.

    Parameters
    ----------
    delegate : object
        The class to get methods/properties & doc-strings.
    accessors : Sequence[str]
        List of accessor to add.
    typ : {'property', 'method'}
    overwrite : bool, default False
       Overwrite the method/property in the target class if it exists.
    accessor_mapping: Callable, default lambda x: x
        Callable to map the delegate's function to the cls' function.
    raise_on_missing: bool, default True
        Raise if an accessor does not exist on delegate.
        False skips the missing accessor.

    Returns
    -------
    callable
        A class decorator.

    Examples
    --------
    @delegate_names(Categorical, ["categories", "ordered"], "property")
    class CategoricalAccessor(PandasDelegate):
        [...]
    """

    def add_delegate_accessors(cls):
        cls._add_delegate_accessors(delegate, accessors, typ, overwrite=overwrite, accessor_mapping=accessor_mapping, raise_on_missing=raise_on_missing)
        return cls
    return add_delegate_accessors