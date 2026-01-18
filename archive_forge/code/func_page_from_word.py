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
def page_from_word(word):
    """
    Return a HTML page for the given word.

    :type word: str
    :param word: The currently active word
    :return: A tuple (page,word), where page is the new current HTML page
        to be sent to the browser and
        word is the new current word
    :rtype: A tuple (str,str)
    """
    return page_from_reference(Reference(word))