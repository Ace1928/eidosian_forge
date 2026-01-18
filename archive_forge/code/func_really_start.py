import errno
import os
import pdb
import socket
import stat
import struct
import sys
import time
import traceback
import gflags as flags
def really_start(main=None):
    """Initializes flag values, and calls main with non-flag arguments.

  Only non-flag arguments are passed to main().  The return value of main() is
  used as the exit status.

  Args:
    main: Main function to run with the list of non-flag arguments, or None
      so that sys.modules['__main__'].main is to be used.
  """
    argv = RegisterAndParseFlagsWithUsage()
    if main is None:
        main = sys.modules['__main__'].main
    try:
        if FLAGS.run_with_pdb:
            sys.exit(pdb.runcall(main, argv))
        elif FLAGS.run_with_profiling:
            import atexit
            if FLAGS.use_cprofile_for_profiling:
                import cProfile as profile
            else:
                import profile
            profiler = profile.Profile()
            atexit.register(profiler.print_stats)
            retval = profiler.runcall(main, argv)
            sys.exit(retval)
        else:
            sys.exit(main(argv))
    except UsageError as error:
        usage(shorthelp=1, detailed_error=error, exitcode=error.exitcode)