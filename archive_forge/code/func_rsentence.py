import cProfile
from io import BytesIO
from html.parser import HTMLParser
import bs4
from bs4 import BeautifulSoup, __version__
from bs4.builder import builder_registry
import os
import pstats
import random
import tempfile
import time
import traceback
import sys
import cProfile
def rsentence(length=4):
    """Generate a random sentence-like string."""
    return ' '.join((rword(random.randint(4, 9)) for i in range(length)))