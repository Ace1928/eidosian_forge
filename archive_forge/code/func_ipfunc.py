import os
def ipfunc():
    """Some ipython tests...

    In [1]: import os

    In [3]: 2+3
    Out[3]: 5

    In [26]: for i in range(3):
       ....:     print(i, end=' ')
       ....:     print(i+1, end=' ')
       ....:
    0 1 1 2 2 3


    It's OK to use '_' for the last result, but do NOT try to use IPython's
    numbered history of _NN outputs, since those won't exist under the
    doctest environment:

    In [7]: 'hi'
    Out[7]: 'hi'

    In [8]: print(repr(_))
    'hi'

    In [7]: 3+4
    Out[7]: 7

    In [8]: _+3
    Out[8]: 10

    In [9]: ipfunc()
    Out[9]: 'ipfunc'
    """
    return 'ipfunc'