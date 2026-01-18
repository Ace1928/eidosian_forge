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
def get_resource_option(self, option_id):
    if option_id in self._resource_option_mapper.keys():
        return self._resource_option_mapper[option_id]
    return None