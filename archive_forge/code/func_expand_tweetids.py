import datetime
import gzip
import itertools
import json
import os
import time
import requests
from twython import Twython, TwythonStreamer
from twython.exceptions import TwythonError, TwythonRateLimitError
from nltk.twitter.api import BasicTweetHandler, TweetHandlerI
from nltk.twitter.util import credsfromfile, guess_path
def expand_tweetids(self, ids_f, verbose=True):
    """
        Given a file object containing a list of Tweet IDs, fetch the
        corresponding full Tweets from the Twitter API.

        The API call `statuses/lookup` will fail to retrieve a Tweet if the
        user has deleted it.

        This call to the Twitter API is rate-limited. See
        <https://dev.twitter.com/rest/reference/get/statuses/lookup> for details.

        :param ids_f: input file object consisting of Tweet IDs, one to a line
        :return: iterable of Tweet objects in JSON format
        """
    ids = [line.strip() for line in ids_f if line]
    if verbose:
        print(f'Counted {len(ids)} Tweet IDs in {ids_f}.')
    id_chunks = [ids[i:i + 100] for i in range(0, len(ids), 100)]
    chunked_tweets = (self.lookup_status(id=chunk) for chunk in id_chunks)
    return itertools.chain.from_iterable(chunked_tweets)