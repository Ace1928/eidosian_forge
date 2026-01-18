from . import _catboost
from .core import Pool, CatBoostError, ARRAY_TYPES, PATH_TYPES, fspath, _update_params_quantize_part, _process_synonyms
from collections import defaultdict
from contextlib import contextmanager
import sys
import numpy as np
import warnings
def read_cd(cd_file, column_count=None, data_file=None, canonize_column_types=False):
    """
    Reads CatBoost column description file
    (see https://catboost.ai/docs/concepts/input-data_column-descfile.html#input-data_column-descfile)

    Parameters
    ----------
    cd_file : str or pathlib.Path
        path to column description file

    column_count : integer
        total number of columns

    data_file : str or pathlib.Path
        path to dataset file in CatBoost format
        specify either column_count directly or data_file to detect it

    canonize_column_types : bool
        if set to True types for columns with synonyms are renamed to canonical type.

    Returns
    -------
    dict with keys:
        "column_type_to_indices" :
            dict of column_type -> column_indices list, column_type is 'Label', 'Categ' etc.

        "column_dtypes" : dict of column_name -> numpy.dtype or 'category'

        "cat_feature_indices" : list of integers
            indices of categorical features in array of all features.
            Note: indices in array of features, not indices in array of all columns!

        "text_feature_indices" : list of integers
            indices of text features in array of all features.
            Note: indices in array of features, not indices in array of all columns!

        "embedding_feature_indices" : list of integers
            indices of embedding features in array of all features.
            Note: indices in array of features, not indices in array of all columns!

        "column_names" : list of strings

        "non_feature_column_indices" : list of integers
    """
    column_type_synonyms_map = {'Target': 'Label', 'DocId': 'SampleId', 'QueryId': 'GroupId'}
    if column_count is None:
        if data_file is None:
            raise Exception('Cannot obtain column count: either specify column_count parameter or specify data_file ' + 'parameter to get it')
        with open(fspath(data_file)) as f:
            column_count = len(f.readline()[:-1].split('\t'))
    column_type_to_indices = {}
    column_dtypes = {}
    cat_feature_indices = []
    text_feature_indices = []
    embedding_feature_indices = []
    column_names = []
    non_feature_column_indices = []
    column_descriptions = []
    with open(fspath(cd_file)) as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if len(line) == 0:
                continue
            line_columns = line.split('\t')
            if len(line_columns) not in [2, 3]:
                raise Exception('Wrong number of columns in cd file')
            column_idx = int(line_columns[0])
            column_type = line_columns[1]
            column_name = None
            if len(line_columns) == 3:
                column_name = line_columns[2]
            column_descriptions.append((column_idx, column_type, column_name))
    column_descriptions.sort()

    def add_missed_columns(start_column_idx, end_column_idx, non_feature_column_count):
        for missed_column_idx in range(start_column_idx, end_column_idx):
            column_name = 'feature_%i' % (missed_column_idx - non_feature_column_count)
            column_names.append(column_name)
            column_type_to_indices.setdefault('Num', []).append(missed_column_idx)
            column_dtypes[column_name] = np.float32
    last_column_idx = -1
    for column_idx, column_type, column_name in column_descriptions:
        if column_idx == last_column_idx:
            raise Exception('Duplicate column indices in cd file')
        add_missed_columns(last_column_idx + 1, column_idx, len(non_feature_column_indices))
        if canonize_column_types:
            column_type = column_type_synonyms_map.get(column_type, column_type)
        column_type_to_indices.setdefault(column_type, []).append(column_idx)
        if column_type in ['Num', 'Categ', 'Text', 'NumVector']:
            feature_idx = column_idx - len(non_feature_column_indices)
            if column_name is None:
                column_name = 'feature_%i' % feature_idx
            if column_type == 'Categ':
                cat_feature_indices.append(feature_idx)
                column_dtypes[column_name] = 'category'
            elif column_type == 'Text':
                text_feature_indices.append(feature_idx)
                column_dtypes[column_name] = object
            elif column_type == 'NumVector':
                embedding_feature_indices.append(feature_idx)
                column_dtypes[column_name] = object
            else:
                column_dtypes[column_name] = np.float32
        else:
            non_feature_column_indices.append(column_idx)
            if column_name is None:
                column_name = column_type
        column_names.append(column_name)
        last_column_idx = column_idx
    add_missed_columns(last_column_idx + 1, column_count, len(non_feature_column_indices))
    return {'column_type_to_indices': column_type_to_indices, 'column_dtypes': column_dtypes, 'cat_feature_indices': cat_feature_indices, 'text_feature_indices': text_feature_indices, 'embedding_feature_indices': embedding_feature_indices, 'column_names': column_names, 'non_feature_column_indices': non_feature_column_indices}