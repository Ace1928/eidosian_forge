import sys
import os
import os.path
import re
import itertools
import warnings
import unicodedata
from docutils import ApplicationError, DataError, __version_info__
from docutils import nodes
from docutils.nodes import unescape
import docutils.io
from docutils.utils.error_reporting import ErrorOutput, SafeString
def normalize_language_tag(tag):
    """Return a list of normalized combinations for a `BCP 47` language tag.

    Example:

    >>> from docutils.utils import normalize_language_tag
    >>> normalize_language_tag('de_AT-1901')
    ['de-at-1901', 'de-at', 'de-1901', 'de']
    >>> normalize_language_tag('de-CH-x_altquot')
    ['de-ch-x-altquot', 'de-ch', 'de-x-altquot', 'de']

    """
    tag = tag.lower().replace('-', '_')
    tag = re.sub('_([a-zA-Z0-9])_', '_\\1-', tag)
    subtags = [subtag for subtag in tag.split('_')]
    base_tag = (subtags.pop(0),)
    taglist = []
    for n in range(len(subtags), 0, -1):
        for tags in itertools.combinations(subtags, n):
            taglist.append('-'.join(base_tag + tags))
    taglist += base_tag
    return taglist