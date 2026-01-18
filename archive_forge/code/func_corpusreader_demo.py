import datetime
import json
from functools import wraps
from io import StringIO
from nltk.twitter import (
@verbose
def corpusreader_demo():
    """
    Use `TwitterCorpusReader` tp read a file of tweets, and print out

    * some full tweets in JSON format;
    * some raw strings from the tweets (i.e., the value of the `text` field); and
    * the result of tokenising the raw strings.

    """
    from nltk.corpus import twitter_samples as tweets
    print()
    print('Complete tweet documents')
    print(SPACER)
    for tweet in tweets.docs('tweets.20150430-223406.json')[:1]:
        print(json.dumps(tweet, indent=1, sort_keys=True))
    print()
    print('Raw tweet strings:')
    print(SPACER)
    for text in tweets.strings('tweets.20150430-223406.json')[:15]:
        print(text)
    print()
    print('Tokenized tweet strings:')
    print(SPACER)
    for toks in tweets.tokenized('tweets.20150430-223406.json')[:15]:
        print(toks)