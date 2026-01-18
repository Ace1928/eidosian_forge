import argparse
import datetime
import json
import os
import re
from pathlib import Path
from typing import Tuple
import yaml
from tqdm import tqdm
from transformers.models.marian.convert_marian_to_pytorch import (
@staticmethod
def model_type_info_from_model_name(name):
    info = {'_has_backtranslated_data': False}
    if '1m' in name:
        info['_data_per_pair'] = str(1000000.0)
    if '2m' in name:
        info['_data_per_pair'] = str(2000000.0)
    if '4m' in name:
        info['_data_per_pair'] = str(4000000.0)
    if '+bt' in name:
        info['_has_backtranslated_data'] = True
    if 'tuned4' in name:
        info['_tuned'] = re.search('tuned4[^-]+', name).group()
    return info