from collections.abc import Mapping
from itertools import chain
import logging
import unittest
import codecs
import os
import os.path
import scipy
import gensim
from gensim.corpora import Dictionary
from gensim.utils import to_utf8
from gensim.test.utils import get_tmpfile, common_texts
def test_from_corpus(self):
    """build `Dictionary` from an existing corpus"""
    documents = ['Human machine interface for lab abc computer applications', 'A survey of user opinion of computer system response time', 'The EPS user interface management system', 'System and human system engineering testing of EPS', 'Relation of user perceived response time to error measurement', 'The generation of random binary unordered trees', 'The intersection graph of paths in trees', 'Graph minors IV Widths of trees and well quasi ordering', 'Graph minors A survey']
    stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
    all_tokens = list(chain.from_iterable(texts))
    tokens_once = set((word for word in set(all_tokens) if all_tokens.count(word) == 1))
    texts = [[word for word in text if word not in tokens_once] for text in texts]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    dictionary_from_corpus = Dictionary.from_corpus(corpus)
    dict_token2id_vals = sorted(dictionary.token2id.values())
    dict_from_corpus_vals = sorted(dictionary_from_corpus.token2id.values())
    self.assertEqual(dict_token2id_vals, dict_from_corpus_vals)
    self.assertEqual(dictionary.dfs, dictionary_from_corpus.dfs)
    self.assertEqual(dictionary.num_docs, dictionary_from_corpus.num_docs)
    self.assertEqual(dictionary.num_pos, dictionary_from_corpus.num_pos)
    self.assertEqual(dictionary.num_nnz, dictionary_from_corpus.num_nnz)
    dictionary_from_corpus_2 = Dictionary.from_corpus(corpus, id2word=dictionary)
    self.assertEqual(dictionary.token2id, dictionary_from_corpus_2.token2id)
    self.assertEqual(dictionary.dfs, dictionary_from_corpus_2.dfs)
    self.assertEqual(dictionary.num_docs, dictionary_from_corpus_2.num_docs)
    self.assertEqual(dictionary.num_pos, dictionary_from_corpus_2.num_pos)
    self.assertEqual(dictionary.num_nnz, dictionary_from_corpus_2.num_nnz)
    bow = gensim.matutils.Sparse2Corpus(scipy.sparse.rand(10, 100))
    dictionary = Dictionary.from_corpus(bow)
    self.assertEqual(dictionary.num_docs, 100)