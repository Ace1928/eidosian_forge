import datetime
import errno
import functools
import hashlib
import json
import os
import re
import sys
import threading
import urllib.parse
import uuid
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
import wrapt
from glanceclient._i18n import _
from glanceclient import exc
def add_req_id_to_object():

    @wrapt.decorator
    def inner(wrapped, instance, args, kwargs):
        return RequestIdProxy(wrapped(*args, **kwargs))
    return inner