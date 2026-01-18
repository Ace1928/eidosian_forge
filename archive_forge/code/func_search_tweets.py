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
def search_tweets(self, keywords, limit=100, lang='en', max_id=None, retries_after_twython_exception=0):
    """
        Call the REST API ``'search/tweets'`` endpoint with some plausible
        defaults. See `the Twitter search documentation
        <https://dev.twitter.com/rest/public/search>`_ for more information
        about admissible search parameters.

        :param str keywords: A list of query terms to search for, written as        a comma-separated string
        :param int limit: Number of Tweets to process
        :param str lang: language
        :param int max_id: id of the last tweet fetched
        :param int retries_after_twython_exception: number of retries when        searching Tweets before raising an exception
        :rtype: python generator
        """
    if not self.handler:
        self.handler = BasicTweetHandler(limit=limit)
    count_from_query = 0
    if max_id:
        self.handler.max_id = max_id
    else:
        results = self.search(q=keywords, count=min(100, limit), lang=lang, result_type='recent')
        count = len(results['statuses'])
        if count == 0:
            print('No Tweets available through REST API for those keywords')
            return
        count_from_query = count
        self.handler.max_id = results['statuses'][count - 1]['id'] - 1
        for result in results['statuses']:
            yield result
            self.handler.counter += 1
            if self.handler.do_continue() == False:
                return
    retries = 0
    while count_from_query < limit:
        try:
            mcount = min(100, limit - count_from_query)
            results = self.search(q=keywords, count=mcount, lang=lang, max_id=self.handler.max_id, result_type='recent')
        except TwythonRateLimitError as e:
            print(f'Waiting for 15 minutes -{e}')
            time.sleep(15 * 60)
            continue
        except TwythonError as e:
            print(f'Fatal error in Twython request -{e}')
            if retries_after_twython_exception == retries:
                raise e
            retries += 1
        count = len(results['statuses'])
        if count == 0:
            print('No more Tweets available through rest api')
            return
        count_from_query += count
        self.handler.max_id = results['statuses'][count - 1]['id'] - 1
        for result in results['statuses']:
            yield result
            self.handler.counter += 1
            if self.handler.do_continue() == False:
                return