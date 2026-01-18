import parlai.core.build_data as build_data
import os
import subprocess
import shutil
import csv
import time
from parlai.core.build_data import DownloadableFile
def divide_csv_into_sets(csv_filepath, sets=('train', 'valid', 'test')):
    reader, fh = read_csv_to_dict_list(csv_filepath)
    base_filename = os.path.basename(csv_filepath).split('.')[0]
    base_path = os.path.dirname(csv_filepath)
    for s in sets:
        path = os.path.join(base_path, base_filename + '_' + s + '.csv')
        fh.seek(0)
        rows = get_rows_for_set(reader, s)
        write_dict_list_to_csv(rows, path)
    fh.close()