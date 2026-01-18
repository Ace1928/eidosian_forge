import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
@property
def content_type(self):
    return self.encoder.content_type