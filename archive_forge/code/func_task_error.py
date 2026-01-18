import re
from collections import Counter
from fileinput import FileInput
import click
from celery.bin.base import CeleryCommand, handle_preload_options
def task_error(self, line, task_name, task_id, result):
    self.task_errors += 1
    if self.on_task_error:
        self.on_task_error(line, task_name, task_id, result)