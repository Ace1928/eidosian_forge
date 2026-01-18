import json
import subprocess
from os.path import join as pjoin
from os.path import isfile
from os.path import isdir
from time import time
from parlai.core.params import ParlaiParser
from data_utils import word_url_tokenize, make_ccid_filter
def select_specific_ids(ccids):
    select = {}
    for i, ccid in enumerate(ccids):
        if not ccid.startswith('<urn:uuid:'):
            ccid = '<urn:uuid:' + ccid
        if not ccid.endswith('>'):
            ccid = ccid + '>'
        select[ccid] = ('specific_ids', i)
    return select