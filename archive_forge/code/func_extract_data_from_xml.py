import parlai.core.build_data as build_data
import glob
import gzip
import multiprocessing
import os
import re
import sys
import time
import tqdm
import xml.etree.ElementTree as ET
from parlai.core.build_data import DownloadableFile
def extract_data_from_xml(xml_object):
    previous_end_time = -1000
    conversation = []
    for sentence_node in xml_object.getroot():
        if sentence_node.tag != 's':
            continue
        words = []
        start_time, end_time = (None, None)
        for node in sentence_node:
            if node.tag == 'time':
                time_value = parse_time_str(node.get('value'))
                if time_value is None:
                    continue
                if node.get('id')[-1] == 'S':
                    start_time = time_value if start_time is None else min(time_value, start_time)
                elif node.get('id')[-1] == 'E':
                    end_time = time_value if end_time is None else max(time_value, end_time)
                else:
                    raise Exception('Unknown time-id for node: %s' % node)
            elif node.tag == 'w':
                if node.text is not None and len(node.text) > 0:
                    words.append(node.text)
            else:
                pass
        sentence = clean_text(words)
        start_time = start_time or previous_end_time
        end_time = end_time or previous_end_time
        if sentence is not None and start_time - previous_end_time <= MAX_TIME_DIFFERENCE_S:
            conversation.append(sentence)
        else:
            if len(conversation) > 1:
                yield conversation
            conversation = []
            if sentence is not None:
                conversation.append(sentence)
        previous_end_time = max(start_time, end_time)