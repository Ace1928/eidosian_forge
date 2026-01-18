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
def generate_reviewer(client, saver, name=NAME):
    """Creates a review producer thread with the given name prefix."""
    real_name = '%s_reviewer' % name
    no_more = threading.Event()
    jb = boards.fetch(real_name, JOBBOARD_CONF, client=client, persistence=saver)

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

    def run():
        """Periodically publishes 'fake' reviews to analyze."""
        jb.connect()
        review_generator = review_iter()
        with contextlib.closing(jb):
            while not no_more.is_set():
                review = next(review_generator)
                details = {'store': {'review': review}}
                job_name = '%s_%s' % (real_name, review['id'])
                print("Posting review '%s'" % review['id'])
                jb.post(job_name, book=make_save_book(saver, review['id']), details=details)
                time.sleep(REVIEW_CREATION_DELAY)
    return (threading_utils.daemon_thread(target=run), no_more.set)