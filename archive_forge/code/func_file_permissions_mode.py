import errno
import io
import os
import pathlib
from urllib.parse import urljoin
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.signals import setting_changed
from django.utils._os import safe_join
from django.utils.deconstruct import deconstructible
from django.utils.encoding import filepath_to_uri
from django.utils.functional import cached_property
from django.utils.timezone import now
from .base import Storage
from .mixins import StorageSettingsMixin
@cached_property
def file_permissions_mode(self):
    return self._value_or_setting(self._file_permissions_mode, settings.FILE_UPLOAD_PERMISSIONS)