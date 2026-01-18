import sqlalchemy
from keystone.common import sql
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
def load_table(self, name):
    table = sqlalchemy.Table(name, sql.ModelBase.metadata, autoload_with=self.database_fixture.engine)
    return table