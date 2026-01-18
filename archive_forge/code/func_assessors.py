from . import schema
from .jsonutil import get_column
from .search import Search
def assessors(self, save=None):
    graph = self.get_graph.rest_resource('assessors')
    self._draw_rest_resource(graph, save=None)