from datetime import datetime, timezone, timedelta
import platform
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.units as munits
from matplotlib.category import UnitData
import numpy as np
import pytest
@pytest.fixture
def quantity_converter():
    qc = munits.ConversionInterface()

    def convert(value, unit, axis):
        if hasattr(value, 'units'):
            return value.to(unit).magnitude
        elif np.iterable(value):
            try:
                return [v.to(unit).magnitude for v in value]
            except AttributeError:
                return [Quantity(v, axis.get_units()).to(unit).magnitude for v in value]
        else:
            return Quantity(value, axis.get_units()).to(unit).magnitude

    def default_units(value, axis):
        if hasattr(value, 'units'):
            return value.units
        elif np.iterable(value):
            for v in value:
                if hasattr(v, 'units'):
                    return v.units
            return None
    qc.convert = MagicMock(side_effect=convert)
    qc.axisinfo = MagicMock(side_effect=lambda u, a: munits.AxisInfo(label=u, default_limits=(0, 100)))
    qc.default_units = MagicMock(side_effect=default_units)
    return qc