import collections
from collections import abc
import itertools
import logging
import re
from oslo_utils import timeutils
import sqlalchemy
from sqlalchemy import Boolean
from sqlalchemy.engine import Connectable
from sqlalchemy.engine import url as sa_url
from sqlalchemy import exc
from sqlalchemy import func
from sqlalchemy import Index
from sqlalchemy import inspect
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy.sql.expression import cast
from sqlalchemy.sql.expression import literal_column
from sqlalchemy.sql import text
from sqlalchemy import Table
from oslo_db._i18n import _
from oslo_db import exception
from oslo_db.sqlalchemy import models
def model_query(model, session, args=None, **kwargs):
    """Query helper for db.sqlalchemy api methods.

    This accounts for `deleted` and `project_id` fields.

    :param model:        Model to query. Must be a subclass of ModelBase.
    :type model:         models.ModelBase

    :param session:      The session to use.
    :type session:       sqlalchemy.orm.session.Session

    :param args:         Arguments to query. If None - model is used.
    :type args:          tuple

    Keyword arguments:

    :keyword project_id: If present, allows filtering by project_id(s).
                         Can be either a project_id value, or an iterable of
                         project_id values, or None. If an iterable is passed,
                         only rows whose project_id column value is on the
                         `project_id` list will be returned. If None is passed,
                         only rows which are not bound to any project, will be
                         returned.
    :type project_id:    iterable,
                         model.__table__.columns.project_id.type,
                         None type

    :keyword deleted:    If present, allows filtering by deleted field.
                         If True is passed, only deleted entries will be
                         returned, if False - only existing entries.
    :type deleted:       bool


    Usage:

    .. code-block:: python

      from oslo_db.sqlalchemy import utils


      def get_instance_by_uuid(uuid):
          session = get_session()
          with session.begin()
              return (utils.model_query(models.Instance, session=session)
                           .filter(models.Instance.uuid == uuid)
                           .first())

      def get_nodes_stat():
          data = (Node.id, Node.cpu, Node.ram, Node.hdd)

          session = get_session()
          with session.begin()
              return utils.model_query(Node, session=session, args=data).all()

    Also you can create your own helper, based on ``utils.model_query()``.
    For example, it can be useful if you plan to use ``project_id`` and
    ``deleted`` parameters from project's ``context``

    .. code-block:: python

      from oslo_db.sqlalchemy import utils


      def _model_query(context, model, session=None, args=None,
                       project_id=None, project_only=False,
                       read_deleted=None):

          # We suppose, that functions ``_get_project_id()`` and
          # ``_get_deleted()`` should handle passed parameters and
          # context object (for example, decide, if we need to restrict a user
          # to query his own entries by project_id or only allow admin to read
          # deleted entries). For return values, we expect to get
          # ``project_id`` and ``deleted``, which are suitable for the
          # ``model_query()`` signature.
          kwargs = {}
          if project_id is not None:
              kwargs['project_id'] = _get_project_id(context, project_id,
                                                     project_only)
          if read_deleted is not None:
              kwargs['deleted'] = _get_deleted_dict(context, read_deleted)
          session = session or get_session()

          with session.begin():
              return utils.model_query(model, session=session,
                                       args=args, **kwargs)

      def get_instance_by_uuid(context, uuid):
          return (_model_query(context, models.Instance, read_deleted='yes')
                        .filter(models.Instance.uuid == uuid)
                        .first())

      def get_nodes_data(context, project_id, project_only='allow_none'):
          data = (Node.id, Node.cpu, Node.ram, Node.hdd)

          return (_model_query(context, Node, args=data, project_id=project_id,
                               project_only=project_only)
                        .all())

    """
    if not issubclass(model, models.ModelBase):
        raise TypeError(_('model should be a subclass of ModelBase'))
    query = session.query(model) if not args else session.query(*args)
    if 'deleted' in kwargs:
        query = _read_deleted_filter(query, model, kwargs['deleted'])
    if 'project_id' in kwargs:
        query = _project_filter(query, model, kwargs['project_id'])
    return query