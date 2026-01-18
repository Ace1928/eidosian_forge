from sqlalchemy import orm
from sqlalchemy import schema
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import update_match
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
def test_custom_handle_failure_cancel_raise(self):
    uuid = '136254d5-3869-408f-9da7-190e0072641a'

    class MyException(Exception):
        pass

    def handle_failure(query):
        result = query.count()
        self.assertEqual(0, result)
        return True
    specimen = MyModel(id=2, y='y1', z='z3', uuid=uuid)
    result = self.session.query(MyModel).update_on_match(specimen, 'uuid', values={'x': 9, 'z': 'z3'}, handle_failure=handle_failure)
    self.assertEqual(uuid, result.uuid)
    self.assertEqual(2, result.id)
    self.assertEqual('z3', result.z)
    self.assertEqual(9, result.x)
    self.assertIn(result, self.session)