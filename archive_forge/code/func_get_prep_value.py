import base64
import pickle
from django.db import models
from django.utils import encoding
import jsonpickle
import oauth2client
def get_prep_value(self, value):
    """Overrides ``models.Field`` method. This is used to convert
        the value from an instances of this class to bytes that can be
        inserted into the database.
        """
    if value is None:
        return None
    else:
        return encoding.smart_text(base64.b64encode(jsonpickle.encode(value).encode()))