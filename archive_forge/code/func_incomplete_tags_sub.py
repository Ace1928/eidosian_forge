import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def incomplete_tags_sub(match):
    return match.group().replace('<', '&lt;')