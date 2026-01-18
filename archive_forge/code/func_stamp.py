from debtcollector import removals
import sqlalchemy
from stevedore import enabled
from oslo_db import exception
def stamp(self, revision):
    """Create stamp for a given revision."""
    return self._plugins[-1].stamp(revision)