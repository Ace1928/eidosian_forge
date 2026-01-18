import xml.sax
import datetime
import itertools
from boto import handler
from boto import config
from boto.mturk.price import Price
import boto.mturk.notification
from boto.connection import AWSQueryConnection
from boto.exception import EC2ResponseError
from boto.resultset import ResultSet
from boto.mturk.question import QuestionForm, ExternalQuestion, HTMLQuestion
@staticmethod
def duration_as_seconds(duration):
    if isinstance(duration, datetime.timedelta):
        duration = duration.days * 86400 + duration.seconds
    try:
        duration = int(duration)
    except TypeError:
        raise TypeError('Duration must be a timedelta or int-castable, got %s' % type(duration))
    return duration