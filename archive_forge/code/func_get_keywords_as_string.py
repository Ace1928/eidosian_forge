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
def get_keywords_as_string(keywords):
    """
        Returns a comma+space-separated string of keywords from either
        a list or a string
        """
    if isinstance(keywords, list):
        keywords = ', '.join(keywords)
    if isinstance(keywords, str):
        final_keywords = keywords
    elif isinstance(keywords, unicode):
        final_keywords = keywords.encode('utf-8')
    elif keywords is None:
        final_keywords = ''
    else:
        raise TypeError('keywords argument must be a string or a list of strings; got a %s' % type(keywords))
    return final_keywords