from . import schema
from .jsonutil import get_column
from .search import Search
def field_values(self, field_name, save=None):
    graph = self.get_graph.field_values(field_name)
    plt.figure(figsize=(8, 8))
    pos = graphviz_layout(graph, prog='twopi', args='')
    cost = lambda v: graph.weights[v]
    graph.weights[field_name] = max([cost(v) for v in graph]) / 2.0
    costs = norm_costs([cost(v) for v in graph], 10000)
    nx.draw(graph, pos, node_size=costs, node_color=costs, font_size=13, font_color='black', font_weight='bold', with_labels=True)
    plt.axis('off')
    if save is not None:
        plt.savefig(save)
    plt.show()