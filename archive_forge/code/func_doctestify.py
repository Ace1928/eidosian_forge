import textwrap
from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.test_other import Test as TestOther
from pyflakes.test.test_imports import Test as TestImports
from pyflakes.test.test_undefined_names import Test as TestUndefinedNames
from pyflakes.test.harness import TestCase, skip
def doctestify(self, input):
    lines = []
    for line in textwrap.dedent(input).splitlines():
        if line.strip() == '':
            pass
        elif line.startswith(' ') or line.startswith('except:') or line.startswith('except ') or line.startswith('finally:') or line.startswith('else:') or line.startswith('elif ') or (lines and lines[-1].startswith(('>>> @', '... @'))):
            line = '... %s' % line
        else:
            line = '>>> %s' % line
        lines.append(line)
    doctestificator = textwrap.dedent('            def doctest_something():\n                """\n                   %s\n                """\n            ')
    return doctestificator % '\n       '.join(lines)