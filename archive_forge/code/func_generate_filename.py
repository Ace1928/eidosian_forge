import datetime
import posixpath
from django import forms
from django.core import checks
from django.core.files.base import File
from django.core.files.images import ImageFile
from django.core.files.storage import Storage, default_storage
from django.core.files.utils import validate_file_name
from django.db.models import signals
from django.db.models.fields import Field
from django.db.models.query_utils import DeferredAttribute
from django.db.models.utils import AltersData
from django.utils.translation import gettext_lazy as _
def generate_filename(self, instance, filename):
    """
        Apply (if callable) or prepend (if a string) upload_to to the filename,
        then delegate further processing of the name to the storage backend.
        Until the storage layer, all file paths are expected to be Unix style
        (with forward slashes).
        """
    if callable(self.upload_to):
        filename = self.upload_to(instance, filename)
    else:
        dirname = datetime.datetime.now().strftime(str(self.upload_to))
        filename = posixpath.join(dirname, filename)
    filename = validate_file_name(filename, allow_relative_path=True)
    return self.storage.generate_filename(filename)