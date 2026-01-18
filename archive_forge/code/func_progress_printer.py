import fractions
import functools
import logging
import os
import string
import sys
import time
from taskflow import engines
from taskflow import exceptions
from taskflow.patterns import linear_flow
from taskflow import task
def progress_printer(task, event_type, details):
    progress = details.pop('progress')
    progress = int(progress * 100.0)
    print("Task '%s' reached %d%% completion" % (task.name, progress))