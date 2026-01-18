import collections
import json
import math
import re
import string
from ...models.bert import BasicTokenizer
from ...utils import logging
def remove_articles(text):
    regex = re.compile('\\b(a|an|the)\\b', re.UNICODE)
    return re.sub(regex, ' ', text)