import os
from io import BytesIO
from django.conf import settings
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from django.utils.module_loading import import_string
def handle_raw_input(self, input_data, META, content_length, boundary, encoding=None):
    """
        Use the content_length to signal whether or not this handler should be
        used.
        """
    self.activated = content_length <= settings.FILE_UPLOAD_MAX_MEMORY_SIZE