import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def occurrence_detector(to_graph, from_graph):
    return sum((1 for node in from_graph.nodes if node in to_graph))