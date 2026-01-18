import random as pyrandom
import time
from functools import partial
from petl.util.random import randomseed, randomtable, RandomTable, dummytable, DummyTable
def test_dummytable_custom_fields():
    """
    Ensure that dummytable provides a table with the right number of rows
    and that it accepts and uses custom column names provided.
    """
    columns = (('count', partial(pyrandom.randint, 0, 100)), ('pet', partial(pyrandom.choice, ['dog', 'cat', 'cow'])), ('color', partial(pyrandom.choice, ['yellow', 'orange', 'brown'])), ('value', pyrandom.random))
    rows = 35
    table = dummytable(numrows=rows, fields=columns)
    assert table[0] == ('count', 'pet', 'color', 'value')
    assert len(table) == rows + 1