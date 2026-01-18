from logging import getLogger
from suds import *
from suds.sudsobject import Object
from suds.sax.element import Element
from suds.sax.date import DateTime, UtcTimezone
from datetime import datetime, timedelta
def setnonceencoding(self, value=False):
    self.nonce_has_encoding = value