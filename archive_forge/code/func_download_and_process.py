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
def download_and_process(file_url, mode, subreddit_names, st_time, output_dir):
    reddit_tmp_dir = pjoin(output_dir, 'reddit_tmp')
    f_name = pjoin(reddit_tmp_dir, file_url.split('/')[-1])
    tries_left = 4
    while tries_left:
        try:
            print('downloading %s %2f' % (f_name, time() - st_time))
            subprocess.run(['wget', '-P', reddit_tmp_dir, file_url], stdout=subprocess.PIPE)
            print('decompressing and filtering %s %2f' % (f_name, time() - st_time))
            if f_name.split('.')[-1] == 'xz':
                f = lzma.open(f_name, 'rt')
            elif f_name.split('.')[-1] == 'bz2':
                f = bz2.open(f_name, 'rt')
            elif f_name.split('.')[-1] == 'zst':
                fh = open(f_name, 'rb')
                dctx = zstd.ZstdDecompressor()
                stream_reader = dctx.stream_reader(fh)
                f = io.TextIOWrapper(stream_reader, encoding='utf-8')
            lines = dict([(name, []) for name in subreddit_names])
            for i, l in enumerate(f):
                if i % 1000000 == 0:
                    print('read %d lines, found %d' % (i, sum([len(ls) for ls in lines.values()])), time() - st_time)
                for name in subreddit_names:
                    subreddit_field = f'"subreddit":"{name}"'
                    if subreddit_field in l:
                        lines[name] += [l.strip()]
            if f_name.split('.')[-1] == 'zst':
                fh.close()
            else:
                f.close()
            os.remove(f_name)
            tries_left = 0
        except EOFError:
            sleep(10)
            print('failed reading file %s file, another %d tries' % (f_name, tries_left))
            os.remove(f_name)
            tries_left -= 1
    print('tokenizing and selecting %s %2f' % (f_name, time() - st_time))
    processed_items = dict([(name, []) for name in subreddit_names])
    if mode == 'submissions':
        key_list = ['id', 'score', 'url', 'title', 'selftext']
    else:
        key_list = ['id', 'link_id', 'parent_id', 'score', 'body']
    for name in subreddit_names:
        for line in lines[name]:
            reddit_dct = json.loads(line)
            if reddit_dct.get('num_comments', 1) > 0 and reddit_dct.get('score', 0) and (reddit_dct.get('score', 0) >= 2) and (mode == 'submissions' or valid_comment(reddit_dct)):
                reddit_res = {}
                for k in key_list:
                    if k in ['title', 'selftext', 'body']:
                        if reddit_dct[k].lower() in ['[removed]', '[deleted]']:
                            reddit_dct[k] = ''
                        txt, url_list = word_url_tokenize(reddit_dct[k])
                        reddit_res[k] = (' '.join(txt.split()), url_list)
                    else:
                        reddit_res[k] = reddit_dct[k]
                processed_items[name] += [reddit_res]
    print('Total found %d' % len(processed_items), time() - st_time)
    return processed_items