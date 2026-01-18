from reportlab.lib import colors
from ._Graph import GraphData
def get_ids(self):
    """Return a list of all ids for the graph set."""
    return list(self._graphs.keys())