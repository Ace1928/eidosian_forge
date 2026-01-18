import json
import subprocess
from os.path import join as pjoin
from os.path import isfile
from os.path import isdir
from time import time
from parlai.core.params import ParlaiParser
from data_utils import word_url_tokenize, make_ccid_filter
def make_docs_directory(output_dir, name):
    if not isdir(pjoin(output_dir, name)):
        subprocess.run(['mkdir', pjoin(output_dir, name)], stdout=subprocess.PIPE)
    for i in range(10):
        if not isdir(pjoin(output_dir, name, str(i))):
            subprocess.run(['mkdir', pjoin(output_dir, name, str(i))], stdout=subprocess.PIPE)