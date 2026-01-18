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
def get_page_range(self, page, total, show=5):
    start = max(page - show // 2, 1)
    stop = min(start + show, total) + 1
    start = max(min(start, stop - show), 1)
    return list(range(start, stop)[:show])