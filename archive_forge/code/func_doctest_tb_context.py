import stack_data
import sys
def doctest_tb_context():
    """
    In [3]: xmode context
    Exception reporting mode: Context

    In [4]: run simpleerr.py
    ---------------------------------------------------------------------------
    ZeroDivisionError                         Traceback (most recent call last)
    <BLANKLINE>
    ...
         30     except IndexError:
         31         mode = 'div'
    ---> 33     bar(mode)
    <BLANKLINE>
    ... in bar(mode)
         15     "bar"
         16     if mode=='div':
    ---> 17         div0()
         18     elif mode=='exit':
         19         try:
    <BLANKLINE>
    ... in div0()
          6     x = 1
          7     y = 0
    ----> 8     x/y
    <BLANKLINE>
    ZeroDivisionError: ..."""