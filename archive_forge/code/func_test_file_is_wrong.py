from pathlib import Path
import pytest
from nltk.corpus import twitter_samples
from nltk.twitter.common import json2csv, json2csv_entities
def test_file_is_wrong(tmp_path, infile):
    """
    Sanity check that file comparison is not giving false positives.
    """
    ref_fn = subdir / 'tweets.20150430-223406.retweet.csv.ref'
    outfn = tmp_path / 'tweets.20150430-223406.text.csv'
    json2csv(infile, outfn, ['text'], gzip_compress=False)
    assert not files_are_identical(outfn, ref_fn)