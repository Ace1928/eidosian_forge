from sqlalchemy import orm
from sqlalchemy import schema
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import update_match
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_instance_criteria_none_list(self):
    specimen = MyModel(y='y1', z=[None])
    self.assertEqual('my_table.y = :y_1 AND my_table.z IS NULL', str(update_match.manufacture_entity_criteria(specimen).compile()))