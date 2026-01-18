import os
import pathlib
from django.core.exceptions import SuspiciousFileOperation
from django.core.files import File
from django.core.files.utils import validate_file_name
from django.utils.crypto import get_random_string
from django.utils.text import get_valid_filename
def get_alternative_name(self, file_root, file_ext):
    """
        Return an alternative filename, by adding an underscore and a random 7
        character alphanumeric string (before the file extension, if one
        exists) to the filename.
        """
    return '%s_%s%s' % (file_root, get_random_string(7), file_ext)