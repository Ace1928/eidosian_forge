from celery.utils.functional import firstmethod, mlazy
from celery.utils.imports import instantiate
def annotate_any(self):
    try:
        return dict(self['*'])
    except KeyError:
        pass