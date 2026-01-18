import pytest
import random
import networkx as nx
from networkx.algorithms import approximation as approx
from networkx.algorithms import threshold
def kernel_integral(u, w, z):
    return z - w