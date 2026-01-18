import sys
import os
import io
import optparse
import re
import unittest
from pprint import pformat
from .test_machinery import PygameTestLoader
def output_into_dots(output):
    """convert the test runner output into dots."""
    verbose_mode = ' ...' in output
    if verbose_mode:
        reasons = {'... ERROR': 'E', '... unexpected success': 'u', '... skipped': 's', '... expected failure': 'x', '... ok': '.', '... FAIL': 'F'}
        results = output.split('\n\n==')[0]
        lines = [l for l in results.split('\n') if l and '...' in l]
        dotlist = []
        for l in lines:
            found = False
            for reason in reasons:
                if reason in l:
                    dotlist.append(reasons[reason])
                    found = True
                    break
            if not found:
                raise ValueError(f'Not sure what this is. Add to reasons. :{l}')
        return ''.join(dotlist)
    dots = DOTS.search(output).group(1)
    return dots