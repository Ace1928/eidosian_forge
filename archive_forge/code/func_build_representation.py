import datetime
from email.utils import quote
import io
import json
import random
import re
import sys
import time
from lazr.uri import URI, merge
from wadllib import (
from wadllib.iso_strptime import iso_strptime
def build_representation(self, media_type=None, param_values=None, **kw_param_values):
    """Build a representation to be sent when invoking this method.

        :return: A 2-tuple of (media_type, representation).
        """
    return self.request.representation(media_type, param_values, **kw_param_values)