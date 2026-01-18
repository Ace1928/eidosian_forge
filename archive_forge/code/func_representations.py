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
@property
def representations(self):
    for definition in self.tag.findall(wadl_xpath('representation')):
        yield RepresentationDefinition(self.application, self.resource, definition)