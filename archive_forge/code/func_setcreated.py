from logging import getLogger
from suds import *
from suds.sudsobject import Object
from suds.sax.element import Element
from suds.sax.date import DateTime, UtcTimezone
from datetime import datetime, timedelta
def setcreated(self, dt=None):
    """
        Set I{created}.
        @param dt: The created date & time.
            Set as datetime.utc() when I{None}.
        @type dt: L{datetime}
        """
    if dt is None:
        self.created = Token.utc()
    else:
        self.created = dt