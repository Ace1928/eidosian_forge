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
def get_unique_keys(model):
    """Get a list of sets of unique model keys.

    :param model: the ORM model class
    :rtype: list of sets of strings
    :return: unique model keys or None if unable to find them
    """
    try:
        mapper = inspect(model)
    except exc.NoInspectionAvailable:
        return None
    else:
        local_table = mapper.local_table
        base_table = mapper.base_mapper.local_table
        if local_table is None:
            return None
    has_info = hasattr(local_table, 'info')
    if has_info:
        info = local_table.info
        if 'oslodb_unique_keys' in info:
            return info['oslodb_unique_keys']
    res = []
    try:
        constraints = base_table.constraints
    except AttributeError:
        constraints = []
    for constraint in constraints:
        if isinstance(constraint, (sqlalchemy.UniqueConstraint, sqlalchemy.PrimaryKeyConstraint)):
            res.append({c.name for c in constraint.columns})
    try:
        indexes = base_table.indexes
    except AttributeError:
        indexes = []
    for index in indexes:
        if index.unique:
            res.append({c.name for c in index.columns})
    if has_info:
        info['oslodb_unique_keys'] = res
    return res