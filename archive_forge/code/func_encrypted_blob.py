from oslo_db import api as oslo_db_api
from sqlalchemy.ext.hybrid import hybrid_property
from keystone.common import driver_hints
from keystone.common import sql
from keystone.credential.backends import base
from keystone import exception
@encrypted_blob.setter
def encrypted_blob(self, encrypted_blob):
    if isinstance(encrypted_blob, bytes):
        encrypted_blob = encrypted_blob.decode('utf-8')
    self._encrypted_blob = encrypted_blob