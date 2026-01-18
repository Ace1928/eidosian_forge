import collections
import contextlib
import copy
import fixtures
import pickle
import sys
from unittest import mock
import warnings
from oslo_config import cfg
from oslo_context import context as oslo_context
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy.orm import registry
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import Table
from oslo_db import exception
from oslo_db import options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import engines as oslo_engines
from oslo_db.sqlalchemy import orm
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
from oslo_db import warning
def test_contexts_picklable(self):
    context = oslo_context.RequestContext()
    with enginefacade.writer.using(context) as session:
        self._assert_ctx_session(context, session)
        pickled = pickle.dumps(context)
        unpickled = pickle.loads(pickled)
        with enginefacade.writer.using(unpickled) as session2:
            self._assert_ctx_session(unpickled, session2)
            assert session is not session2