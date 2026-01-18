import json
import subprocess
from os.path import join as pjoin
from os.path import isfile
from os.path import isdir
from time import time
from parlai.core.params import ParlaiParser
from data_utils import word_url_tokenize, make_ccid_filter
def check_url(select, url):
    if url.startswith('http://') or url.startswith('https://'):
        url = url.split('//', 1)[-1]
    return select.get(url, False)