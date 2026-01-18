import itertools
import os
import warnings
import pytest
import networkx as nx
def test_draw_shell_nlist():
    try:
        nlist = [list(range(4)), list(range(4, 10)), list(range(10, 14))]
        nx.draw_shell(barbell, nlist=nlist)
        plt.savefig('test.ps')
    finally:
        try:
            os.unlink('test.ps')
        except OSError:
            pass