import importlib
import pkgutil
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import urlopen
import pytest
import modin.pandas
from modin.utils import PANDAS_API_URL_TEMPLATE
def test_all_urls_exist(doc_urls):
    broken = []
    methods_with_broken_urls = ('pandas.DataFrame.flags', 'pandas.Series.info', 'pandas.DataFrame.isetitem', 'pandas.Series.swapaxes', 'pandas.DataFrame.to_numpy', 'pandas.Series.axes', 'pandas.Series.divmod', 'pandas.Series.rdivmod')
    for broken_method in methods_with_broken_urls:
        doc_urls.remove(PANDAS_API_URL_TEMPLATE.format(broken_method))

    def _test_url(url):
        try:
            with urlopen(url):
                pass
        except HTTPError:
            broken.append(url)
    with ThreadPoolExecutor(32) as pool:
        pool.map(_test_url, doc_urls)
    assert not broken, 'Invalid URLs detected'