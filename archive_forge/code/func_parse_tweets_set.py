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
def parse_tweets_set(filename, label, word_tokenizer=None, sent_tokenizer=None, skip_header=True):
    """
    Parse csv file containing tweets and output data a list of (text, label) tuples.

    :param filename: the input csv filename.
    :param label: the label to be appended to each tweet contained in the csv file.
    :param word_tokenizer: the tokenizer instance that will be used to tokenize
        each sentence into tokens (e.g. WordPunctTokenizer() or BlanklineTokenizer()).
        If no word_tokenizer is specified, tweets will not be tokenized.
    :param sent_tokenizer: the tokenizer that will be used to split each tweet into
        sentences.
    :param skip_header: if True, skip the first line of the csv file (which usually
        contains headers).

    :return: a list of (text, label) tuples.
    """
    tweets = []
    if not sent_tokenizer:
        sent_tokenizer = load('tokenizers/punkt/english.pickle')
    with codecs.open(filename, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        if skip_header == True:
            next(reader, None)
        i = 0
        for tweet_id, text in reader:
            i += 1
            sys.stdout.write(f'Loaded {i} tweets\r')
            if word_tokenizer:
                tweet = [w for sent in sent_tokenizer.tokenize(text) for w in word_tokenizer.tokenize(sent)]
            else:
                tweet = text
            tweets.append((tweet, label))
    print(f'Loaded {i} tweets')
    return tweets