from . import schema
from .jsonutil import get_column
from .search import Search
def datatypes(self, pattern='*', save=None):
    graph = self.get_graph.datatypes(pattern)
    plt.figure(figsize=(8, 8))
    pos = graphviz_layout(graph, prog='twopi', args='')
    cost = lambda v: float(graph.degree(v)) ** 3 + graph.weights[v] ** 2
    node_size = [cost(v) for v in graph]
    node_color = [cost(v) for v in graph]
    nx.draw(graph, pos, node_size=node_size, node_color=node_color, font_size=13, font_color='green', font_weight='bold', with_labels=True)
    plt.axis('off')
    if save is not None:
        plt.savefig(save)
    plt.show()