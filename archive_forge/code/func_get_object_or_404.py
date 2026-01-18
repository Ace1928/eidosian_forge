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
def get_object_or_404(query_or_model, *query):
    if not isinstance(query_or_model, SelectQuery):
        query_or_model = query_or_model.select()
    try:
        return query_or_model.where(*query).get()
    except DoesNotExist:
        abort(404)