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
def benchmark_parsers(num_elements=100000):
    """Very basic head-to-head performance benchmark."""
    print('Comparative parser benchmark on Beautiful Soup %s' % __version__)
    data = rdoc(num_elements)
    print('Generated a large invalid HTML document (%d bytes).' % len(data))
    for parser in ['lxml', ['lxml', 'html'], 'html5lib', 'html.parser']:
        success = False
        try:
            a = time.time()
            soup = BeautifulSoup(data, parser)
            b = time.time()
            success = True
        except Exception as e:
            print('%s could not parse the markup.' % parser)
            traceback.print_exc()
        if success:
            print('BS4+%s parsed the markup in %.2fs.' % (parser, b - a))
    from lxml import etree
    a = time.time()
    etree.HTML(data)
    b = time.time()
    print('Raw lxml parsed the markup in %.2fs.' % (b - a))
    import html5lib
    parser = html5lib.HTMLParser()
    a = time.time()
    parser.parse(data)
    b = time.time()
    print('Raw html5lib parsed the markup in %.2fs.' % (b - a))