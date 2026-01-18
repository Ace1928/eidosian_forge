import csv
import gzip
import json
from nltk.internals import deprecated
def json2csv(fp, outfile, fields, encoding='utf8', errors='replace', gzip_compress=False):
    """
    Extract selected fields from a file of line-separated JSON tweets and
    write to a file in CSV format.

    This utility function allows a file of full tweets to be easily converted
    to a CSV file for easier processing. For example, just TweetIDs or
    just the text content of the Tweets can be extracted.

    Additionally, the function allows combinations of fields of other Twitter
    objects (mainly the users, see below).

    For Twitter entities (e.g. hashtags of a Tweet), and for geolocation, see
    `json2csv_entities`

    :param str infile: The name of the file containing full tweets

    :param str outfile: The name of the text file where results should be    written

    :param list fields: The list of fields to be extracted. Useful examples    are 'id_str' for the tweetID and 'text' for the text of the tweet. See    <https://dev.twitter.com/overview/api/tweets> for a full list of fields.    e. g.: ['id_str'], ['id', 'text', 'favorite_count', 'retweet_count']    Additionally, it allows IDs from other Twitter objects, e. g.,    ['id', 'text', 'user.id', 'user.followers_count', 'user.friends_count']

    :param error: Behaviour for encoding errors, see    https://docs.python.org/3/library/codecs.html#codec-base-classes

    :param gzip_compress: if `True`, output files are compressed with gzip
    """
    writer, outf = _outf_writer(outfile, encoding, errors, gzip_compress)
    writer.writerow(fields)
    for line in fp:
        tweet = json.loads(line)
        row = extract_fields(tweet, fields)
        writer.writerow(row)
    outf.close()