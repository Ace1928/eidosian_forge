from reportlab.lib import colors
from ._Graph import GraphData
def del_graph(self, graph_id):
    """Remove a graph from the set, indicated by its id."""
    del self._graphs[graph_id]