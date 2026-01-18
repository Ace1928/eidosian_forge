import stack_data
import sys
def doctest_tb_sysexit_verbose_stack_data_05():
    """
        In [18]: %run simpleerr.py exit
        An exception has occurred, use %tb to see the full traceback.
        SystemExit: (1, 'Mode = exit')

        In [19]: %run simpleerr.py exit 2
        An exception has occurred, use %tb to see the full traceback.
        SystemExit: (2, 'Mode = exit')

        In [23]: %xmode verbose
        Exception reporting mode: Verbose

        In [24]: %tb
        ---------------------------------------------------------------------------
        SystemExit                                Traceback (most recent call last)
        <BLANKLINE>
        ...
            30     except IndexError:
            31         mode = 'div'
        ---> 33     bar(mode)
                mode = 'exit'
        <BLANKLINE>
        ... in bar(mode='exit')
            ...     except:
            ...         stat = 1
        ---> ...     sysexit(stat, mode)
                mode = 'exit'
                stat = 2
            ...     else:
            ...         raise ValueError('Unknown mode')
        <BLANKLINE>
        ... in sysexit(stat=2, mode='exit')
            10 def sysexit(stat, mode):
        ---> 11     raise SystemExit(stat, f"Mode = {mode}")
                stat = 2
        <BLANKLINE>
        SystemExit: (2, 'Mode = exit')
        """