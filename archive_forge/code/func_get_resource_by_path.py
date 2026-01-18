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
def get_resource_by_path(self, path):
    """Locate one of the resources described by this document.

        :param path: The path to the resource.
        """
    matching = [resource for resource in self.resources if resource.attrib['path'] == path]
    if len(matching) < 1:
        return None
    if len(matching) > 1:
        raise WADLError('More than one resource defined with path %s' % path)
    return Resource(self, merge(self.resource_base, path, True), matching[0])