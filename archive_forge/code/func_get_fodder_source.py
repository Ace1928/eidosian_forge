import difflib
import inspect
import re
import unittest
from code import compile_command as compiler
from functools import partial
from bpython.curtsiesfrontend.interpreter import code_finished_will_parse
from bpython.curtsiesfrontend.preprocess import preprocess
from bpython.test.fodder import original, processed
def get_fodder_source(test_name):
    pattern = f'#StartTest-{test_name}\\n(.*?)#EndTest'
    orig, xformed = [re.search(pattern, inspect.getsource(module), re.DOTALL) for module in [original, processed]]
    if not orig:
        raise ValueError(f"Can't locate test {test_name} in original fodder file")
    if not xformed:
        raise ValueError(f"Can't locate test {test_name} in processed fodder file")
    return (orig.group(1), xformed.group(1))