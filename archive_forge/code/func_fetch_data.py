from numpy import array, frombuffer, load
from ._registry import registry, registry_urls
def fetch_data(dataset_name, data_fetcher=data_fetcher):
    if data_fetcher is None:
        raise ImportError("Missing optional dependency 'pooch' required for scipy.datasets module. Please use pip or conda to install 'pooch'.")
    return data_fetcher.fetch(dataset_name)