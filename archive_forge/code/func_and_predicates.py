import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
def and_predicates(predicates):
    """Return a predicate that is true iff all predicates are true."""
    return lambda *args, **kwargs: all((p(*args, **kwargs) for p in predicates))