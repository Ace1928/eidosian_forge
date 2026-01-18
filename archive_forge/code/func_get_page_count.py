import math
import sys
from flask import abort
from flask import render_template
from flask import request
from peewee import Database
from peewee import DoesNotExist
from peewee import Model
from peewee import Proxy
from peewee import SelectQuery
from playhouse.db_url import connect as db_url_connect
def get_page_count(self):
    if not hasattr(self, '_page_count'):
        self._page_count = int(math.ceil(float(self.query.count()) / self.paginate_by))
    return self._page_count