import inspect
from importlib import import_module
from celery._state import get_current_app
from celery.app.autoretry import add_autoretry_behaviour
from celery.exceptions import InvalidTaskError, NotRegistered
def filter_types(self, type):
    return {name: task for name, task in self.items() if getattr(task, 'type', 'regular') == type}