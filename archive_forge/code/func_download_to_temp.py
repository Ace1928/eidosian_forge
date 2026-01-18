import contextlib
from pathlib import Path
from unittest import mock
import pytest
import cartopy
import cartopy.io as cio
from cartopy.io.shapereader import NEShpDownloader
@pytest.fixture
def download_to_temp(tmp_path_factory):
    """
    Context manager which defaults the "data_dir" to a temporary directory
    which is automatically cleaned up on exit.

    """
    with contextlib.ExitStack() as stack:
        stack.callback(cartopy.config.__setitem__, 'downloaders', cartopy.config['downloaders'].copy())
        stack.callback(cartopy.config.__setitem__, 'data_dir', cartopy.config['data_dir'])
        tmp_dir = tmp_path_factory.mktemp('cartopy_data')
        cartopy.config['data_dir'] = str(tmp_dir)
        yield tmp_dir