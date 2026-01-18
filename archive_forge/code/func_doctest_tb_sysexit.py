import stack_data
import sys
def doctest_tb_sysexit():
    """
    In [17]: %xmode plain
    Exception reporting mode: Plain

    In [18]: %run simpleerr.py exit
    An exception has occurred, use %tb to see the full traceback.
    SystemExit: (1, 'Mode = exit')

    In [19]: %run simpleerr.py exit 2
    An exception has occurred, use %tb to see the full traceback.
    SystemExit: (2, 'Mode = exit')

    In [20]: %tb
    Traceback (most recent call last):
      File ...:... in execfile
        exec(compiler(f.read(), fname, "exec"), glob, loc)
      File ...:...
        bar(mode)
      File ...:... in bar
        sysexit(stat, mode)
      File ...:... in sysexit
        raise SystemExit(stat, f"Mode = {mode}")
    SystemExit: (2, 'Mode = exit')

    In [21]: %xmode context
    Exception reporting mode: Context

    In [22]: %tb
    ---------------------------------------------------------------------------
    SystemExit                                Traceback (most recent call last)
    File ..., in execfile(fname, glob, loc, compiler)
         ... with open(fname, "rb") as f:
         ...     compiler = compiler or compile
    ---> ...     exec(compiler(f.read(), fname, "exec"), glob, loc)
    ...
         30     except IndexError:
         31         mode = 'div'
    ---> 33     bar(mode)
    <BLANKLINE>
    ...bar(mode)
         21         except:
         22             stat = 1
    ---> 23         sysexit(stat, mode)
         24     else:
         25         raise ValueError('Unknown mode')
    <BLANKLINE>
    ...sysexit(stat, mode)
         10 def sysexit(stat, mode):
    ---> 11     raise SystemExit(stat, f"Mode = {mode}")
    <BLANKLINE>
    SystemExit: (2, 'Mode = exit')
    """