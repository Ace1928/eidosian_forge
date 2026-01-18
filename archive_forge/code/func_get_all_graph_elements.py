import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def get_all_graph_elements(graph, l=None):
    """Return all nodes and edges, including elements in subgraphs"""
    if not l:
        l = []
        outer = True
        l.append(graph)
    else:
        outer = False
    for element in graph.allitems:
        if isinstance(element, dotparsing.DotSubGraph):
            l.append(element)
            get_all_graph_elements(element, l)
        else:
            l.append(element)
    if outer:
        return l
    else:
        l.append(EndOfGraphElement())