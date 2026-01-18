import codecs
import csv
import json
import pickle
import random
import re
import sys
import time
from copy import deepcopy
import nltk
from nltk.corpus import CategorizedPlaintextCorpusReader
from nltk.data import load
from nltk.tokenize.casual import EMOTICON_RE
def output_markdown(filename, **kwargs):
    """
    Write the output of an analysis to a file.
    """
    with codecs.open(filename, 'at') as outfile:
        text = '\n*** \n\n'
        text += '{} \n\n'.format(time.strftime('%d/%m/%Y, %H:%M'))
        for k in sorted(kwargs):
            if isinstance(kwargs[k], dict):
                dictionary = kwargs[k]
                text += f'  - **{k}:**\n'
                for entry in sorted(dictionary):
                    text += f'    - {entry}: {dictionary[entry]} \n'
            elif isinstance(kwargs[k], list):
                text += f'  - **{k}:**\n'
                for entry in kwargs[k]:
                    text += f'    - {entry}\n'
            else:
                text += f'  - **{k}:** {kwargs[k]} \n'
        outfile.write(text)