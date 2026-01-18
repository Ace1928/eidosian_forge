import random as pyrandom
import time
from functools import partial
from petl.util.random import randomseed, randomtable, RandomTable, dummytable, DummyTable
def test_dummytable_class():
    """
    Ensure that DummyTable provides a table with the right number of rows
    and columns.
    """
    rows = 70
    table = DummyTable(numrows=rows)
    assert len(table) == rows + 1