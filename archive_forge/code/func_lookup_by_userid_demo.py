import datetime
import json
from functools import wraps
from io import StringIO
from nltk.twitter import (
@verbose
def lookup_by_userid_demo():
    """
    Use the REST API to convert a userID to a screen name.
    """
    oauth = credsfromfile()
    client = Query(**oauth)
    user_info = client.user_info_from_id(USERIDS)
    for info in user_info:
        name = info['screen_name']
        followers = info['followers_count']
        following = info['friends_count']
        print(f'{name}, followers: {followers}, following: {following}')