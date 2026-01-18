import json
from peewee import *
from peewee import Expression
from peewee import Node
from peewee import NodeList
from playhouse.postgres_ext import ArrayField
from playhouse.postgres_ext import DateTimeTZField
from playhouse.postgres_ext import IndexedFieldMixin
from playhouse.postgres_ext import IntervalField
from playhouse.postgres_ext import Match
from playhouse.postgres_ext import TSVectorField
from playhouse.postgres_ext import _JsonLookupBase
def is_connection_usable(self):
    if self._state.closed:
        return False
    conn = self._state.conn
    return conn.pgconn.transaction_status < conn.TransactionStatus.INERROR