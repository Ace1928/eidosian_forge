from matplotlib import cbook, units
import matplotlib.dates as date_ticker
@staticmethod
def epoch2float(value, unit):
    """
        Convert an Epoch value to a float suitable for plotting as a python
        datetime object.

        = INPUT VARIABLES
        - value    An Epoch or list of Epochs that need to be converted.
        - unit     The units to use for an axis with Epoch data.

        = RETURN VALUE
        - Returns the value parameter converted to floats.
        """
    return value.julianDate(unit) - EpochConverter.jdRef