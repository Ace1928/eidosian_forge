import io
import pyomo.common.unittest as unittest
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
def test_latexPrinter_fileWriter(self):
    m = generate_simple_model()
    with TempfileManager.new_context() as tempfile:
        fd, fname = tempfile.mkstemp()
        pstr = latex_printer(m, ostream=fname)
        f = open(fname)
        bstr = f.read()
        f.close()
        bstr_split = bstr.split('\n')
        bstr_stripped = bstr_split[8:-2]
        bstr = '\n'.join(bstr_stripped) + '\n'
        self.assertEqual(pstr + '\n', bstr)
    self.assertRaises(ValueError, latex_printer, **{'pyomo_component': m, 'ostream': 2.0})