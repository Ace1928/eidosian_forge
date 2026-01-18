import os
import pickle
import re
from xml.etree import ElementTree as ET
from nltk.tag import ClassifierBasedTagger, pos_tag
from nltk.chunk.api import ChunkParserI
from nltk.chunk.util import ChunkScore
from nltk.data import find
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
def load_ace_data(roots, fmt='binary', skip_bnews=True):
    for root in roots:
        for root, dirs, files in os.walk(root):
            if root.endswith('bnews') and skip_bnews:
                continue
            for f in files:
                if f.endswith('.sgm'):
                    yield from load_ace_file(os.path.join(root, f), fmt)