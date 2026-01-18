from sklearn.datasets import load_breast_cancer
from ray import tune
from ray.data import read_datasource, Dataset, Datasource, ReadTask
from ray.data.block import BlockMetadata
from ray.tune.impl.utils import execute_dataset
def load_data():
    data_raw = load_breast_cancer(as_frame=True)
    dataset_df = data_raw['data']
    dataset_df['target'] = data_raw['target']
    return [pa.Table.from_pandas(dataset_df)]