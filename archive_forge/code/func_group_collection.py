from datetime import datetime, timedelta, timezone
from kombu.exceptions import EncodeError
from kombu.utils.objects import cached_property
from kombu.utils.url import maybe_sanitize_url, urlparse
from celery import states
from celery.exceptions import ImproperlyConfigured
from .base import BaseBackend
@cached_property
def group_collection(self):
    """Get the meta-data task collection."""
    collection = self.database[self.groupmeta_collection]
    collection.create_index('date_done', background=True)
    return collection