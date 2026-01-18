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
def get_unique_counter_from_url(sp):
    """
    Extract the unique counter from the URL if it has one.  Otherwise return
    null.
    """
    pos = sp.rfind('%23')
    if pos != -1:
        return int(sp[pos + 3:])
    else:
        return None