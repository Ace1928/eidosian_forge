import numpy as np
from ase.gui.i18n import _
import warnings
def getresult(name, get_quantity):
    try:
        if calc.calculation_required(atoms, [name]):
            quantity = None
        else:
            quantity = get_quantity()
    except Exception as err:
        quantity = None
        errmsg = 'An error occurred while retrieving {} from the calculator: {}'.format(name, err)
        warnings.warn(errmsg)
    return quantity