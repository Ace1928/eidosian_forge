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
def get_object_list(self):
    if self.check_bounds and self.get_page() > self.get_page_count():
        abort(404)
    return self.query.paginate(self.get_page(), self.paginate_by)