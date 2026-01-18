import warnings
import numpy as np
from ase.optimize.optimize import Dynamics
from ase.md.logger import MDLogger
from ase.io.trajectory import Trajectory
from ase import units
def process_temperature(temperature, temperature_K, orig_unit):
    """Handle that temperature can be specified in multiple units.

    For at least a transition period, molecular dynamics in ASE can
    have the temperature specified in either Kelvin or Electron
    Volt.  The different MD algorithms had different defaults, by
    forcing the user to explicitly choose a unit we can resolve
    this.  Using the original method then will issue a
    FutureWarning.

    Four parameters:

    temperature: None or float
        The original temperature specification in whatever unit was
        historically used.  A warning is issued if this is not None and
        the historical unit was eV.

    temperature_K: None or float
        Temperature in Kelvin.

    orig_unit: str
        Unit used for the `temperature`` parameter.  Must be 'K' or 'eV'.

    Exactly one of the two temperature parameters must be different from 
    None, otherwise an error is issued.

    Return value: Temperature in Kelvin.
    """
    if (temperature is not None) + (temperature_K is not None) != 1:
        raise TypeError("Exactly one of the parameters 'temperature'," + " and 'temperature_K', must be given")
    if temperature is not None:
        w = "Specify the temperature in K using the 'temperature_K' argument"
        if orig_unit == 'K':
            return temperature
        elif orig_unit == 'eV':
            warnings.warn(FutureWarning(w))
            return temperature / units.kB
        else:
            raise ValueError('Unknown temperature unit ' + orig_unit)
    assert temperature_K is not None
    return temperature_K