def doctest_multiline3():
    """Multiline examples with blank lines.

    In [12]: def h(x):
       ....:     if x>1:
       ....:         return x**2
       ....:     # To leave a blank line in the input, you must mark it
       ....:     # with a comment character:
       ....:     #
       ....:     # otherwise the doctest parser gets confused.
       ....:     else:
       ....:         return -1
       ....:      

    In [13]: h(5)
    Out[13]: 25

    In [14]: h(1)
    Out[14]: -1

    In [15]: h(0)
    Out[15]: -1
   """