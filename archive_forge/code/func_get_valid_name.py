import os
import pathlib
from django.core.exceptions import SuspiciousFileOperation
from django.core.files import File
from django.core.files.utils import validate_file_name
from django.utils.crypto import get_random_string
from django.utils.text import get_valid_filename
def get_valid_name(self, name):
    """
        Return a filename, based on the provided filename, that's suitable for
        use in the target storage system.
        """
    return get_valid_filename(name)