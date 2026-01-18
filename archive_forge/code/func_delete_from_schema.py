import fixtures
import os
import testresources
import testscenarios
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
def delete_from_schema(self, engine):
    """A hook which should delete all data from an existing schema.

        Should *not* drop any objects, just remove data from tables
        that needs to be reset between tests.
        """