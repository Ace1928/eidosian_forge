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
def linked_resource(self):
    """Follow a link from this parameter to a new resource.

        This only works for parameters whose WADL definition includes a
        <link> tag that points to a known WADL description.

        :return: A Resource object for the resource at the other end
        of the link.
        """
    link = self.link
    if link is None:
        raise ValueError("This parameter isn't a link to anything.")
    return link.follow