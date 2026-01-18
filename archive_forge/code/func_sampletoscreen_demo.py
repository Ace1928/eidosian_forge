import datetime
import json
from functools import wraps
from io import StringIO
from nltk.twitter import (
@verbose
def sampletoscreen_demo(limit=20):
    """
    Sample from the Streaming API and send output to terminal.
    """
    oauth = credsfromfile()
    client = Streamer(**oauth)
    client.register(TweetViewer(limit=limit))
    client.sample()