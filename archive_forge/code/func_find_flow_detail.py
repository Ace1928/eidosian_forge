import contextlib
import hashlib
import logging
import os
import random
import sys
import time
from oslo_utils import uuidutils
from taskflow import engines
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import models
from taskflow import task
import example_utils  # noqa
def find_flow_detail(backend, book_id, flow_id):
    with contextlib.closing(backend.get_connection()) as conn:
        lb = conn.get_logbook(book_id)
        return lb.find(flow_id)