import abc
import keystone.conf
from keystone import exception
@property
def is_sql(self):
    """Indicate if this Driver uses SQL."""
    return False