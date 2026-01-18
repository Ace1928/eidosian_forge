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
def demo_tweets(trainer, n_instances=None, output=None):
    """
    Train and test Naive Bayes classifier on 10000 tweets, tokenized using
    TweetTokenizer.
    Features are composed of:

    - 1000 most frequent unigrams
    - 100 top bigrams (using BigramAssocMeasures.pmi)

    :param trainer: `train` method of a classifier.
    :param n_instances: the number of total tweets that have to be used for
        training and testing. Tweets will be equally split between positive and
        negative.
    :param output: the output file where results have to be reported.
    """
    from nltk.corpus import stopwords, twitter_samples
    from nltk.sentiment import SentimentAnalyzer
    from nltk.tokenize import TweetTokenizer
    tokenizer = TweetTokenizer(preserve_case=False)
    if n_instances is not None:
        n_instances = int(n_instances / 2)
    fields = ['id', 'text']
    positive_json = twitter_samples.abspath('positive_tweets.json')
    positive_csv = 'positive_tweets.csv'
    json2csv_preprocess(positive_json, positive_csv, fields, limit=n_instances)
    negative_json = twitter_samples.abspath('negative_tweets.json')
    negative_csv = 'negative_tweets.csv'
    json2csv_preprocess(negative_json, negative_csv, fields, limit=n_instances)
    neg_docs = parse_tweets_set(negative_csv, label='neg', word_tokenizer=tokenizer)
    pos_docs = parse_tweets_set(positive_csv, label='pos', word_tokenizer=tokenizer)
    train_pos_docs, test_pos_docs = split_train_test(pos_docs)
    train_neg_docs, test_neg_docs = split_train_test(neg_docs)
    training_tweets = train_pos_docs + train_neg_docs
    testing_tweets = test_pos_docs + test_neg_docs
    sentim_analyzer = SentimentAnalyzer()
    all_words = [word for word in sentim_analyzer.all_words(training_tweets)]
    unigram_feats = sentim_analyzer.unigram_word_feats(all_words, top_n=1000)
    sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
    bigram_collocs_feats = sentim_analyzer.bigram_collocation_feats([tweet[0] for tweet in training_tweets], top_n=100, min_freq=12)
    sentim_analyzer.add_feat_extractor(extract_bigram_feats, bigrams=bigram_collocs_feats)
    training_set = sentim_analyzer.apply_features(training_tweets)
    test_set = sentim_analyzer.apply_features(testing_tweets)
    classifier = sentim_analyzer.train(trainer, training_set)
    try:
        classifier.show_most_informative_features()
    except AttributeError:
        print('Your classifier does not provide a show_most_informative_features() method.')
    results = sentim_analyzer.evaluate(test_set)
    if output:
        extr = [f.__name__ for f in sentim_analyzer.feat_extractors]
        output_markdown(output, Dataset='labeled_tweets', Classifier=type(classifier).__name__, Tokenizer=tokenizer.__class__.__name__, Feats=extr, Results=results, Instances=n_instances)