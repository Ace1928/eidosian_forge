import textwrap
import pycodestyle
from keystone.tests.hacking import checks
from keystone.tests import unit
from keystone.tests.unit.ksfixtures import hacking as hacking_fixtures
def test_for_translations(self):
    for example in self.code_ex.examples:
        code = self.code_ex.shared_imports + example['code']
        errors = example['expected_errors']
        self.assert_has_errors(code, expected_errors=errors)