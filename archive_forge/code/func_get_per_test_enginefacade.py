import fixtures
import os
import testresources
import testscenarios
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
def get_per_test_enginefacade(self):
    """Return an enginefacade._TransactionContextManager per test.

        This facade should be the one that the test expects the code to
        use.   Usually this is the same one returned by get_engineafacade()
        which is the default.  For special applications like Gnocchi,
        this can be overridden to provide an instance-level facade.

        """
    return self.get_enginefacade()