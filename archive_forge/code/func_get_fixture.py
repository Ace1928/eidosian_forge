import textwrap
import pycodestyle
from keystone.tests.hacking import checks
from keystone.tests import unit
from keystone.tests.unit.ksfixtures import hacking as hacking_fixtures
def get_fixture(self):
    return hacking_fixtures.HackingTranslations()