import abc
import logging
import os
import random
import re
import string
import sqlalchemy
from sqlalchemy import schema
from sqlalchemy import sql
import testresources
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import utils
def provisioned_database_url(self, base_url, ident):
    if base_url.database:
        return utils.make_url('sqlite:////tmp/%s.db' % ident)
    else:
        return base_url