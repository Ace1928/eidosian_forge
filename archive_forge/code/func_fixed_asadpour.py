import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def fixed_asadpour(G, weight):
    return nx_app.asadpour_atsp(G, weight, 56)