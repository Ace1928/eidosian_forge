from matplotlib import cbook, units
import matplotlib.dates as date_ticker
@staticmethod
def float2epoch(value, unit):
    """
        Convert a Matplotlib floating-point date into an Epoch of the specified
        units.

        = INPUT VARIABLES
        - value     The Matplotlib floating-point date.
        - unit      The unit system to use for the Epoch.

        = RETURN VALUE
        - Returns the value converted to an Epoch in the specified time system.
        """
    import matplotlib.testing.jpl_units as U
    secPastRef = value * 86400.0 * U.UnitDbl(1.0, 'sec')
    return U.Epoch(unit, secPastRef, EpochConverter.jdRef)