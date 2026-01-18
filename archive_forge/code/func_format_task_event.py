import sys
from datetime import datetime
from celery.app import app_or_default
from celery.utils.functional import LRUCache
from celery.utils.time import humanize_seconds
def format_task_event(self, hostname, timestamp, type, task, event):
    fields = ', '.join((f'{key}={event[key]}' for key in sorted(event)))
    sep = fields and ':' or ''
    self.say(f'{hostname} [{timestamp}] {humanize_type(type)}{sep} {task} {fields}')