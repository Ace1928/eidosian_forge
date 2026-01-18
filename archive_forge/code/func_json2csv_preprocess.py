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
def json2csv_preprocess(json_file, outfile, fields, encoding='utf8', errors='replace', gzip_compress=False, skip_retweets=True, skip_tongue_tweets=True, skip_ambiguous_tweets=True, strip_off_emoticons=True, remove_duplicates=True, limit=None):
    """
    Convert json file to csv file, preprocessing each row to obtain a suitable
    dataset for tweets Semantic Analysis.

    :param json_file: the original json file containing tweets.
    :param outfile: the output csv filename.
    :param fields: a list of fields that will be extracted from the json file and
        kept in the output csv file.
    :param encoding: the encoding of the files.
    :param errors: the error handling strategy for the output writer.
    :param gzip_compress: if True, create a compressed GZIP file.

    :param skip_retweets: if True, remove retweets.
    :param skip_tongue_tweets: if True, remove tweets containing ":P" and ":-P"
        emoticons.
    :param skip_ambiguous_tweets: if True, remove tweets containing both happy
        and sad emoticons.
    :param strip_off_emoticons: if True, strip off emoticons from all tweets.
    :param remove_duplicates: if True, remove tweets appearing more than once.
    :param limit: an integer to set the number of tweets to convert. After the
        limit is reached the conversion will stop. It can be useful to create
        subsets of the original tweets json data.
    """
    with codecs.open(json_file, encoding=encoding) as fp:
        writer, outf = _outf_writer(outfile, encoding, errors, gzip_compress)
        writer.writerow(fields)
        if remove_duplicates == True:
            tweets_cache = []
        i = 0
        for line in fp:
            tweet = json.loads(line)
            row = extract_fields(tweet, fields)
            try:
                text = row[fields.index('text')]
                if skip_retweets == True:
                    if re.search('\\bRT\\b', text):
                        continue
                if skip_tongue_tweets == True:
                    if re.search('\\:\\-?P\\b', text):
                        continue
                if skip_ambiguous_tweets == True:
                    all_emoticons = EMOTICON_RE.findall(text)
                    if all_emoticons:
                        if set(all_emoticons) & HAPPY and set(all_emoticons) & SAD:
                            continue
                if strip_off_emoticons == True:
                    row[fields.index('text')] = re.sub('(?!\\n)\\s+', ' ', EMOTICON_RE.sub('', text))
                if remove_duplicates == True:
                    if row[fields.index('text')] in tweets_cache:
                        continue
                    else:
                        tweets_cache.append(row[fields.index('text')])
            except ValueError:
                pass
            writer.writerow(row)
            i += 1
            if limit and i >= limit:
                break
        outf.close()