from io import StringIO, TextIOWrapper
from email.feedparser import FeedParser, BytesFeedParser
from email._policybase import compat32
def parsebytes(self, text, headersonly=True):
    return BytesParser.parsebytes(self, text, headersonly=True)