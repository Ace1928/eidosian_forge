from unittest import TestCase
from ipykernel.tests import utils
from nbformat.converter import convert
from nbformat.reader import reads
import re
import json
from copy import copy
import unittest
def scrub_outputs(self, outputs):
    """
        remove all scrubs from output data and text
        """
    for output in outputs:
        out = copy(output)
        for scrub, sub in []:

            def _scrubLines(lines):
                if isstr(lines):
                    return re.sub(scrub, sub, lines)
                else:
                    return [re.sub(scrub, sub, line) for line in lines]
            if 'text' in out:
                out['text'] = _scrubLines(out['text'])
            if 'data' in out:
                if isinstance(out['data'], dict):
                    for mime, data in out['data'].items():
                        out['data'][mime] = _scrubLines(data)
                else:
                    out['data'] = _scrubLines(out['data'])
        yield out