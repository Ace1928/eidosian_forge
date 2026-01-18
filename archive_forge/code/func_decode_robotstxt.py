import logging
import sys
from abc import ABCMeta, abstractmethod
from scrapy.utils.python import to_unicode
def decode_robotstxt(robotstxt_body, spider, to_native_str_type=False):
    try:
        if to_native_str_type:
            robotstxt_body = to_unicode(robotstxt_body)
        else:
            robotstxt_body = robotstxt_body.decode('utf-8')
    except UnicodeDecodeError:
        logger.warning('Failure while parsing robots.txt. File either contains garbage or is in an encoding other than UTF-8, treating it as an empty file.', exc_info=sys.exc_info(), extra={'spider': spider})
        robotstxt_body = ''
    return robotstxt_body