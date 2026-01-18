import parlai.core.build_data as build_data
import os
import subprocess
import shutil
import csv
import time
from parlai.core.build_data import DownloadableFile
def get_rows_for_set(reader, req_set):
    selected_rows = [row for row in reader if row['set'].strip() == req_set]
    return selected_rows