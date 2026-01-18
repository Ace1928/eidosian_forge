import datetime
import email
from io import StringIO
from urllib.parse import urlparse
from django.utils.encoding import iri_to_uri
from django.utils.xmlutils import SimplerXMLGenerator
def latest_post_date(self):
    """
        Return the latest item's pubdate or updateddate. If no items
        have either of these attributes this return the current UTC date/time.
        """
    latest_date = None
    date_keys = ('updateddate', 'pubdate')
    for item in self.items:
        for date_key in date_keys:
            item_date = item.get(date_key)
            if item_date:
                if latest_date is None or item_date > latest_date:
                    latest_date = item_date
    return latest_date or datetime.datetime.now(tz=datetime.timezone.utc)