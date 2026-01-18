from reportlab.lib import colors
from ._Graph import GraphData
def get_graphs(self):
    """Return list of all graphs in the graph set, sorted by id.

        Sorting is to ensure reliable stacking.
        """
    return [self._graphs[id] for id in sorted(self._graphs)]