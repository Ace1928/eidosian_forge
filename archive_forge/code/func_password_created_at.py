import datetime
import sqlalchemy
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import orm
from sqlalchemy.orm import collections
from keystone.common import password_hashing
from keystone.common import resource_options
from keystone.common import sql
import keystone.conf
from keystone.identity.backends import resource_options as iro
@property
def password_created_at(self):
    """Return when password was created at."""
    if self.password_ref:
        return self.password_ref.created_at
    return None