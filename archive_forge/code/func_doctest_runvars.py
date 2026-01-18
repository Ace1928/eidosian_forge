import inspect
def doctest_runvars():
    """Test that variables defined in scripts get loaded correctly via %run.

    In [13]: run simplevars.py
    x is: 1

    In [14]: x
    Out[14]: 1
    """