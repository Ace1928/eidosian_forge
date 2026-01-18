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
def cmp_chunks(correct, guessed):
    correct = NEChunkParser._parse_to_tagged(correct)
    guessed = NEChunkParser._parse_to_tagged(guessed)
    ellipsis = False
    for (w, ct), (w, gt) in zip(correct, guessed):
        if ct == gt == 'O':
            if not ellipsis:
                print(f'  {ct:15} {gt:15} {w}')
                print('  {:15} {:15} {2}'.format('...', '...', '...'))
                ellipsis = True
        else:
            ellipsis = False
            print(f'  {ct:15} {gt:15} {w}')