from __future__ import with_statement, division
import logging
import unittest
import os
from collections import namedtuple
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import doc2vec, keyedvectors
from gensim.test.utils import datapath, get_tmpfile, temporary_file, common_texts as raw_sentences
def read_su_sentiment_rotten_tomatoes(dirname, lowercase=True):
    """
    Read and return documents from the Stanford Sentiment Treebank
    corpus (Rotten Tomatoes reviews), from http://nlp.Stanford.edu/sentiment/

    Initialize the corpus from a given directory, where
    http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
    has been expanded. It's not too big, so compose entirely into memory.
    """
    logging.info('loading corpus from %s', dirname)
    chars_sst_mangled = ['à', 'á', 'â', 'ã', 'æ', 'ç', 'è', 'é', 'í', 'í', 'ï', 'ñ', 'ó', 'ô', 'ö', 'û', 'ü']
    sentence_fixups = [(char.encode('utf-8').decode('latin1'), char) for char in chars_sst_mangled]
    sentence_fixups.extend([('Â', ''), ('\xa0', ' '), ('-LRB-', '('), ('-RRB-', ')')])
    phrase_fixups = [('\xa0', ' ')]
    info_by_sentence = {}
    with open(os.path.join(dirname, 'datasetSentences.txt'), 'r') as sentences:
        with open(os.path.join(dirname, 'datasetSplit.txt'), 'r') as splits:
            next(sentences)
            next(splits)
            for sentence_line, split_line in zip(sentences, splits):
                id, text = sentence_line.split('\t')
                id = int(id)
                text = text.rstrip()
                for junk, fix in sentence_fixups:
                    text = text.replace(junk, fix)
                id2, split_i = split_line.split(',')
                assert id == int(id2)
                if text not in info_by_sentence:
                    info_by_sentence[text] = (id, int(split_i))
    phrases = [None] * 239232
    with open(os.path.join(dirname, 'dictionary.txt'), 'r') as phrase_lines:
        for line in phrase_lines:
            text, id = line.split('|')
            for junk, fix in phrase_fixups:
                text = text.replace(junk, fix)
            phrases[int(id)] = text.rstrip()
    SentimentPhrase = namedtuple('SentimentPhrase', SentimentDocument._fields + ('sentence_id',))
    with open(os.path.join(dirname, 'sentiment_labels.txt'), 'r') as sentiments:
        next(sentiments)
        for line in sentiments:
            id, sentiment = line.split('|')
            id = int(id)
            sentiment = float(sentiment)
            text = phrases[id]
            words = text.split()
            if lowercase:
                words = [word.lower() for word in words]
            sentence_id, split_i = info_by_sentence.get(text, (None, 0))
            split = [None, 'train', 'test', 'dev'][split_i]
            phrases[id] = SentimentPhrase(words, [id], split, sentiment, sentence_id)
    assert sum((1 for phrase in phrases if phrase.sentence_id is not None)) == len(info_by_sentence)
    assert sum((1 for phrase in phrases if phrase.split == 'train')) == 8531
    assert sum((1 for phrase in phrases if phrase.split == 'test')) == 2210
    assert sum((1 for phrase in phrases if phrase.split == 'dev')) == 1100
    logging.info('loaded corpus with %i sentences and %i phrases from %s', len(info_by_sentence), len(phrases), dirname)
    return phrases