import contextlib
from pathlib import Path
from unittest import mock
import pytest
import cartopy
import cartopy.io as cio
from cartopy.io.shapereader import NEShpDownloader
@pytest.mark.network
def test_downloading_simple_ascii(download_to_temp):
    file_url = 'https://ajax.googleapis.com/ajax/libs/jquery/1.8.2/{name}.js'
    format_dict = {'name': 'jquery'}
    target_template = str(download_to_temp / '{name}.txt')
    tmp_fname = Path(target_template.format(**format_dict))
    dnld_item = cio.Downloader(file_url, target_template)
    assert dnld_item.target_path(format_dict) == tmp_fname
    with pytest.warns(cio.DownloadWarning):
        assert dnld_item.path(format_dict) == tmp_fname
    with open(tmp_fname) as fh:
        fh.readline()
        assert fh.readline() == ' * jQuery JavaScript Library v1.8.2\n'
    with mock.patch.object(dnld_item, 'acquire_resource', wraps=dnld_item.acquire_resource) as counter:
        assert dnld_item.path(format_dict) == tmp_fname
    counter.assert_not_called()