from matplotlib import cbook, units
import matplotlib.dates as date_ticker
@staticmethod
def axisinfo(unit, axis):
    majloc = date_ticker.AutoDateLocator()
    majfmt = date_ticker.AutoDateFormatter(majloc)
    return units.AxisInfo(majloc=majloc, majfmt=majfmt, label=unit)