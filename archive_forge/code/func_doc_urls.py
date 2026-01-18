import importlib
import pkgutil
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import urlopen
import pytest
import modin.pandas
from modin.utils import PANDAS_API_URL_TEMPLATE
@pytest.fixture
def doc_urls(get_generated_doc_urls):
    for modinfo in pkgutil.walk_packages(modin.pandas.__path__, 'modin.pandas.'):
        try:
            importlib.import_module(modinfo.name)
        except ModuleNotFoundError:
            pass
    return sorted(get_generated_doc_urls())