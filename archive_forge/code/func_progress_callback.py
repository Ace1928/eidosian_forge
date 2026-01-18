from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def progress_callback(*args, **kwargs):
    raise Exception('Woot!')