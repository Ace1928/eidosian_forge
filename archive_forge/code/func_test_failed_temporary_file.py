import os
import pathlib
import tempfile
import numpy as np
import pytest
from skimage import io
from skimage._shared.testing import assert_array_equal, fetch
from skimage.data import data_dir
@pytest.mark.parametrize('error_class', [FileNotFoundError, FileExistsError, PermissionError, BaseException])
def test_failed_temporary_file(monkeypatch, error_class):
    fetch('data/camera.png')
    data_path = data_dir.lstrip(os.path.sep)
    data_path = data_path.replace(os.path.sep, '/')
    image_url = f'file:///{data_path}/camera.png'
    with monkeypatch.context():
        monkeypatch.setattr(tempfile, 'NamedTemporaryFile', _named_tempfile_func(error_class))
        with pytest.raises(error_class):
            io.imread(image_url)