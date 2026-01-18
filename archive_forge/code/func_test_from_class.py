import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_from_class(self):
    """New context returned with lineno updated from class"""
    path = 'cls.py'

    class A:
        pass

    class B:
        pass
    cls_lines = {'A': 5, 'B': 7}
    context = export_pot._ModuleContext(path, _source_info=(cls_lines, {}))
    contextA = context.from_class(A)
    self.check_context(contextA, path, 5)
    contextB1 = context.from_class(B)
    self.check_context(contextB1, path, 7)
    contextB2 = contextA.from_class(B)
    self.check_context(contextB2, path, 7)
    self.check_context(context, path, 1)
    self.assertEqual('', self.get_log())