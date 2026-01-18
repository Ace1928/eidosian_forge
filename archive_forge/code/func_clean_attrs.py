import os
import tempfile
import networkx as nx
def clean_attrs(which, added):
    for attr in added:
        del G.graph[which][attr]
    if not G.graph[which]:
        del G.graph[which]