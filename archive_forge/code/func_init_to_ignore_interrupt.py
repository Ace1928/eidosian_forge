import bz2
import logging
import multiprocessing
import re
import signal
from pickle import PicklingError
from xml.etree.ElementTree import iterparse
from gensim import utils
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus
def init_to_ignore_interrupt():
    """Enables interruption ignoring.

    Warnings
    --------
    Should only be used when master is prepared to handle termination of
    child processes.

    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)