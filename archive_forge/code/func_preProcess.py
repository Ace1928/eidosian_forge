import warnings
import numpy
def preProcess(self, front):
    """Sets up the list data structure needed for calculation."""
    dimensions = len(self.referencePoint)
    nodeList = _MultiList(dimensions)
    nodes = [_MultiList.Node(dimensions, point) for point in front]
    for i in range(dimensions):
        self.sortByDimension(nodes, i)
        nodeList.extend(nodes, i)
    self.list = nodeList