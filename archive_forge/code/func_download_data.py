import os
from typing import Callable, Dict
import numpy as np
from transformers import EvalPrediction
from transformers import glue_compute_metrics, glue_output_modes
def download_data(task_name, data_dir='./data'):
    print('Downloading dataset.')
    import urllib
    import zipfile
    if task_name == 'rte':
        url = 'https://dl.fbaipublicfiles.com/glue/data/RTE.zip'
    else:
        raise ValueError('Unknown task: {}'.format(task_name))
    data_file = os.path.join(data_dir, '{}.zip'.format(task_name))
    if not os.path.exists(data_file):
        urllib.request.urlretrieve(url, data_file)
        with zipfile.ZipFile(data_file) as zip_ref:
            zip_ref.extractall(data_dir)
        print('Downloaded data for task {} to {}'.format(task_name, data_dir))
    else:
        print('Data already exists. Using downloaded data for task {} from {}'.format(task_name, data_dir))