import copy
from .chemistry import Substance
from .units import (
from .util.arithmeticdict import ArithmeticDict, _imul, _itruediv
from .printing import as_per_substance_html_table
@classmethod
def of_quantity(cls, quantity_name, *args, **kwargs):
    instance = cls(get_derived_unit(SI_base_registry, quantity_name), *args, **kwargs)
    instance.quantity_name = quantity_name
    return instance