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
def parse_time_str(time_value_str):
    if not (time_value_str is not None and len(time_value_str) == 12 and (time_value_str[2] == ':') and (time_value_str[5] == ':') and (time_value_str[8] == ',')):
        return None
    try:
        return int(time_value_str[0:2]) * 3600 + int(time_value_str[3:5]) * 60 + int(time_value_str[6:8])
    except ValueError:
        return None