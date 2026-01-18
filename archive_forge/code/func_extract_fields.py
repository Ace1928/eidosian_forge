import csv
import gzip
import json
from nltk.internals import deprecated
def extract_fields(tweet, fields):
    """
    Extract field values from a full tweet and return them as a list

    :param json tweet: The tweet in JSON format
    :param list fields: The fields to be extracted from the tweet
    :rtype: list(str)
    """
    out = []
    for field in fields:
        try:
            _add_field_to_out(tweet, field, out)
        except TypeError as e:
            raise RuntimeError('Fatal error when extracting fields. Cannot find field ', field) from e
    return out