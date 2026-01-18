import os
from io import BytesIO
from django.conf import settings
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from django.utils.module_loading import import_string
def file_complete(self, file_size):
    """Return a file object if this handler is activated."""
    if not self.activated:
        return
    self.file.seek(0)
    return InMemoryUploadedFile(file=self.file, field_name=self.field_name, name=self.file_name, content_type=self.content_type, size=file_size, charset=self.charset, content_type_extra=self.content_type_extra)