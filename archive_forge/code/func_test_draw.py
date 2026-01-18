import itertools
import os
import warnings
import pytest
import networkx as nx
def test_draw():
    try:
        functions = [nx.draw_circular, nx.draw_kamada_kawai, nx.draw_planar, nx.draw_random, nx.draw_spectral, nx.draw_spring, nx.draw_shell]
        options = [{'node_color': 'black', 'node_size': 100, 'width': 3}]
        for function, option in itertools.product(functions, options):
            function(barbell, **option)
            plt.savefig('test.ps')
    finally:
        try:
            os.unlink('test.ps')
        except OSError:
            pass