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
def connect_db(self):
    if self._excluded_routes and request.endpoint in self._excluded_routes:
        return
    self.database.connect()