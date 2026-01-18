import sqlalchemy
from keystone.common import sql
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
def assertExpectedSchema(self, table, cols):
    table = self.load_table(table)
    for col, type_, length in cols:
        self.assertIsInstance(table.c[col].type, type_)
        if length:
            self.assertEqual(length, table.c[col].type.length)