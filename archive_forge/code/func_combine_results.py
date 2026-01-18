import sys
import os
import io
import optparse
import re
import unittest
from pprint import pformat
from .test_machinery import PygameTestLoader
def combine_results(all_results, t):
    """

    Return pieced together results in a form fit for human consumption. Don't
    rely on results if  piecing together subprocessed  results (single process
    mode is fine). Was originally meant for that  purpose but was found to be
    unreliable.  See the dump option for reliable results.

    """
    all_dots = ''
    failures = []
    for module, results in sorted(all_results.items()):
        output, return_code, raw_return = map(results.get, ('output', 'return_code', 'raw_return'))
        if not output or (return_code and RAN_TESTS_DIV not in output):
            output_lines = raw_return.splitlines()
            if len(output_lines) > 20:
                results['raw_return'] = '\n'.join(output_lines[:10] + ['...'] + output_lines[-10:])
            failures.append(COMPLETE_FAILURE_TEMPLATE % results)
            all_dots += 'E'
            continue
        dots = output_into_dots(output)
        all_dots += dots
        tracebacks = extract_tracebacks(output)
        if tracebacks:
            failures.append(tracebacks)
    total_fails, total_errors = map(all_dots.count, 'FE')
    total_tests = len(all_dots)
    combined = [all_dots]
    if failures:
        combined += [''.join(failures).lstrip('\n')[:-1]]
    combined += [f'{RAN_TESTS_DIV} {total_tests} tests in {t:.3f}s\n']
    if failures:
        infos = ([f'failures={total_fails}'] if total_fails else []) + ([f'errors={total_errors}'] if total_errors else [])
        combined += [f'FAILED ({', '.join(infos)})\n']
    else:
        combined += ['OK\n']
    return (total_tests, '\n'.join(combined))