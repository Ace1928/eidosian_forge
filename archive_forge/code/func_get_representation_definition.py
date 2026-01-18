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
def get_representation_definition(self, media_type):
    """Get one of the possible representations of the response."""
    if self.tag is None:
        return None
    for representation in self:
        if representation.media_type == media_type:
            return representation
    return None