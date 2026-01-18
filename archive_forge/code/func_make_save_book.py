import contextlib
import itertools
import logging
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
from oslo_utils import timeutils
from oslo_utils import uuidutils
from zake import fake_client
from taskflow.conductors import backends as conductors
from taskflow import engines
from taskflow.jobs import backends as boards
from taskflow.patterns import linear_flow
from taskflow.persistence import backends as persistence
from taskflow.persistence import models
from taskflow import task
from taskflow.utils import threading_utils
def make_save_book(saver, review_id):
    book = models.LogBook('book_%s' % review_id)
    detail = models.FlowDetail('flow_%s' % review_id, uuidutils.generate_uuid())
    book.add(detail)
    factory_args = ()
    factory_kwargs = {}
    engines.save_factory_details(detail, create_review_workflow, factory_args, factory_kwargs)
    with contextlib.closing(saver.get_connection()) as conn:
        conn.save_logbook(book)
        return book