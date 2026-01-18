import json
import re
import logging
def find_last_failed(self, argv):
    from numba.tests.support import captured_output
    listargv = ['-l'] + [a for a in argv if not a.startswith('-')]
    with captured_output('stdout') as stream:
        main(*listargv)
        pat = re.compile('^(\\w+\\.)+\\w+$')
        lines = stream.getvalue().splitlines()
    all_tests = [x for x in lines if pat.match(x) is not None]
    try:
        fobj = open(self.cache_filename)
    except OSError:
        failed_tests = []
    else:
        with fobj as fin:
            failed_tests = json.load(fin)
    return (all_tests, failed_tests)