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
def get_static_index_page(with_shutdown):
    """
    Get the static index page.
    """
    template = '\n<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Frameset//EN"  "http://www.w3.org/TR/html4/frameset.dtd">\n<HTML>\n     <!-- Natural Language Toolkit: Wordnet Interface: Graphical Wordnet Browser\n            Copyright (C) 2001-2023 NLTK Project\n            Author: Jussi Salmela <jtsalmela@users.sourceforge.net>\n            URL: <https://www.nltk.org/>\n            For license information, see LICENSE.TXT -->\n     <HEAD>\n         <TITLE>NLTK Wordnet Browser</TITLE>\n     </HEAD>\n\n<frameset rows="7%%,93%%">\n    <frame src="%s" name="header">\n    <frame src="start_page" name="body">\n</frameset>\n</HTML>\n'
    if with_shutdown:
        upper_link = 'upper.html'
    else:
        upper_link = 'upper_2.html'
    return template % upper_link