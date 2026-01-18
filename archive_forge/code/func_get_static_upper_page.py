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
def get_static_upper_page(with_shutdown):
    """
    Return the upper frame page,

    If with_shutdown is True then a 'shutdown' button is also provided
    to shutdown the server.
    """
    template = '\n<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">\n<html>\n    <!-- Natural Language Toolkit: Wordnet Interface: Graphical Wordnet Browser\n        Copyright (C) 2001-2023 NLTK Project\n        Author: Jussi Salmela <jtsalmela@users.sourceforge.net>\n        URL: <https://www.nltk.org/>\n        For license information, see LICENSE.TXT -->\n    <head>\n                <meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1" />\n        <title>Untitled Document</title>\n    </head>\n    <body>\n    <form method="GET" action="search" target="body">\n            Current Word:&nbsp;<input type="text" id="currentWord" size="10" disabled>\n            Next Word:&nbsp;<input type="text" id="nextWord" name="nextWord" size="10">\n            <input name="searchButton" type="submit" value="Search">\n    </form>\n        <a target="body" href="web_help.html">Help</a>\n        %s\n\n</body>\n</html>\n'
    if with_shutdown:
        shutdown_link = '<a href="SHUTDOWN THE SERVER">Shutdown</a>'
    else:
        shutdown_link = ''
    return template % shutdown_link