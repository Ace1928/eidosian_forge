from html.entities import codepoint2name
from collections import defaultdict
import codecs
import re
import logging
import string
from html.entities import html5
def chardet_dammit(s):
    return None