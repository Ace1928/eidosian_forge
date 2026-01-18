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
def representation(self, media_type=None, param_values=None, **kw_param_values):
    """Build a representation to be sent along with this request.

        :return: A 2-tuple of (media_type, representation).
        """
    definition = self.get_representation_definition(media_type)
    if definition is None:
        raise TypeError('Cannot build representation of media type %s' % media_type)
    return definition.bind(param_values, **kw_param_values)