from warnings import simplefilter
import numpy as np
from sklearn.utils.multiclass import unique_labels
import wandb
from wandb.sklearn import utils
def make_columns(class_ids, counts_train, counts_test):
    class_column, dataset_column, count_column = ([], [], [])
    for i in range(len(class_ids)):
        class_column.append(class_ids[i])
        dataset_column.append('train')
        count_column.append(counts_train[i])
        class_column.append(class_ids[i])
        dataset_column.append('test')
        count_column.append(counts_test[i])
        if utils.check_against_limit(i, 'class_proportions', utils.chart_limit):
            break
    return (class_column, dataset_column, count_column)