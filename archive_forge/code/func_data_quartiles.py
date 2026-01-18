from reportlab.lib import colors
from ._Graph import GraphData
def data_quartiles(self):
    """Return (minimum, lowerQ, medianQ, upperQ, maximum) values as a tuple."""
    data = []
    for graph in self._graphs.values():
        data += list(graph.data.values())
    data.sort()
    datalen = len(data)
    return (data[0], data[datalen / 4], data[datalen / 2], data[3 * datalen / 4], data[-1])