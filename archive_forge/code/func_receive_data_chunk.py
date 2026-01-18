import os
from io import BytesIO
from django.conf import settings
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from django.utils.module_loading import import_string
def receive_data_chunk(self, raw_data, start):
    """Add the data to the BytesIO file."""
    if self.activated:
        self.file.write(raw_data)
    else:
        return raw_data