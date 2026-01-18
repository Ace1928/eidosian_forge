import bz2
import io
import json
import lzma
import os
import re
import requests
import subprocess
import zstandard as zstd
from bs4 import BeautifulSoup
from os.path import isfile
from os.path import join as pjoin
from time import sleep, time
from collections import defaultdict
from parlai.core.params import ParlaiParser
from data_utils import word_url_tokenize
def get_comments_from_post(post_id):
    comment_ids_link = f'{API_URL}submission/comment_ids/{post_id}'
    p_req = requests.get(comment_ids_link)
    comment_json = json.loads(p_req.text)
    cids = comment_json['data']
    comment_link = f'{API_URL}comment/search?ids='
    for i in range(0, len(cids), 100):
        curr_comments = ','.join(cids[i:i + 100])
        curr_comments_link = comment_link + curr_comments
        c_req = requests.get(curr_comments_link)
        comments = json.loads(c_req.text)
        for comment in comments['data']:
            yield (comment['subreddit'].lower(), json.dumps(comment))