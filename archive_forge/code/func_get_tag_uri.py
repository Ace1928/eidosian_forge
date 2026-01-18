import datetime
import email
from io import StringIO
from urllib.parse import urlparse
from django.utils.encoding import iri_to_uri
from django.utils.xmlutils import SimplerXMLGenerator
def get_tag_uri(url, date):
    """
    Create a TagURI.

    See
    https://web.archive.org/web/20110514113830/http://diveintomark.org/archives/2004/05/28/howto-atom-id
    """
    bits = urlparse(url)
    d = ''
    if date is not None:
        d = ',%s' % date.strftime('%Y-%m-%d')
    return 'tag:%s%s:%s/%s' % (bits.hostname, d, bits.path, bits.fragment)