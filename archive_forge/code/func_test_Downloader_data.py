import contextlib
from pathlib import Path
from unittest import mock
import pytest
import cartopy
import cartopy.io as cio
from cartopy.io.shapereader import NEShpDownloader
def test_Downloader_data():
    di = cio.Downloader('https://testing.com/{category}/{name}.zip', str(Path('{data_dir}') / '{category}' / 'shape.shp'), str(Path('/project') / 'foobar' / '{category}' / 'sample.shp'))
    replacement_dict = {'category': 'example', 'name': 'test', 'data_dir': str(Path('/wibble') / 'foo' / 'bar')}
    assert di.url(replacement_dict) == 'https://testing.com/example/test.zip'
    assert di.target_path(replacement_dict) == Path('/wibble') / 'foo' / 'bar' / 'example' / 'shape.shp'
    assert di.pre_downloaded_path(replacement_dict) == Path('/project/foobar/example/sample.shp')