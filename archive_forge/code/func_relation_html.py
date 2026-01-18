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
def relation_html(r):
    if isinstance(r, Synset):
        return make_lookup_link(Reference(r.lemma_names()[0]), r.lemma_names()[0])
    elif isinstance(r, Lemma):
        return relation_html(r.synset())
    elif isinstance(r, tuple):
        return '{}\n<ul>{}</ul>\n'.format(relation_html(r[0]), ''.join(('<li>%s</li>\n' % relation_html(sr) for sr in r[1])))
    else:
        raise TypeError('r must be a synset, lemma or list, it was: type(r) = %s, r = %s' % (type(r), r))