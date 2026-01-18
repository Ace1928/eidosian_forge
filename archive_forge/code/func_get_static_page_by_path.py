import base64
import copy
import getopt
import io
import os
import pickle
import sys
import threading
import time
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from sys import argv
from urllib.parse import unquote_plus
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Lemma, Synset
def get_static_page_by_path(path):
    """
    Return a static HTML page from the path given.
    """
    if path == 'index_2.html':
        return get_static_index_page(False)
    elif path == 'index.html':
        return get_static_index_page(True)
    elif path == 'NLTK Wordnet Browser Database Info.html':
        return 'Display of Wordnet Database Statistics is not supported'
    elif path == 'upper_2.html':
        return get_static_upper_page(False)
    elif path == 'upper.html':
        return get_static_upper_page(True)
    elif path == 'web_help.html':
        return get_static_web_help_page()
    elif path == 'wx_help.html':
        return get_static_wx_help_page()
    raise FileNotFoundError()