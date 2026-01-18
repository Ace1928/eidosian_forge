import datetime
import json
from functools import wraps
from io import StringIO
from nltk.twitter import (
@verbose
def streamtofile_demo(limit=20):
    """
    Write 20 tweets sampled from the public Streaming API to a file.
    """
    oauth = credsfromfile()
    client = Streamer(**oauth)
    client.register(TweetWriter(limit=limit, repeat=False))
    client.statuses.sample()