from reportlab.lib import colors
from math import sqrt
def quartiles(self):
    """Return (minimum, lowerQ, medianQ, upperQ, maximum) values as tuple."""
    data = sorted(self.data.values())
    datalen = len(data)
    return (data[0], data[datalen // 4], data[datalen // 2], data[3 * datalen // 4], data[-1])