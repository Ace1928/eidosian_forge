from __future__ import annotations
import cProfile
import pstats
import signal
import sys
import tempfile
from collections import namedtuple
from functools import partial, wraps
from typing import Any, Callable, Union
def prof_main(main):
    """
    Decorator for profiling main programs.

    Profiling is activated by prepending the command line options
    supported by the original main program with the keyword `prof`.

    Examples:

            $ script.py arg --foo=1

        becomes

            $ script.py prof arg --foo=1

        The decorated main accepts two new arguments:

            prof_file: Name of the output file with profiling data
                If not given, a temporary file is created.
            sortby: Profiling data are sorted according to this value.
                default is "time". See sort_stats.
    """

    @wraps(main)
    def wrapper(*args, **kwargs):
        try:
            do_prof = sys.argv[1] == 'prof'
            if do_prof:
                sys.argv.pop(1)
        except Exception:
            do_prof = False
        if not do_prof:
            sys.exit(main())
        else:
            print('Entering profiling mode...')
            prof_file = kwargs.get('prof_file', None)
            if prof_file is None:
                _, prof_file = tempfile.mkstemp()
                print(f'Profiling data stored in {prof_file}')
            sortby = kwargs.get('sortby', 'time')
            cProfile.runctx('main()', globals(), locals(), prof_file)
            s = pstats.Stats(prof_file)
            s.strip_dirs().sort_stats(sortby).print_stats()
            if 'retval' not in kwargs:
                sys.exit(0)
            else:
                return kwargs['retval']
    return wrapper