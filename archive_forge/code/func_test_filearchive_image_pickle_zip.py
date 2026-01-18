import json
import os
import tarfile
import zipfile
import numpy as np
from holoviews import Image
from holoviews.core.io import FileArchive, Serializer
def test_filearchive_image_pickle_zip(self, tmp_path):
    export_name = 'archive_image'
    filenames = ['Group1-Im1.hvz', 'Group2-Im2.hvz']
    archive = FileArchive(root=os.fspath(tmp_path), export_name=export_name, pack=True, archive_format='zip')
    archive.add(self.image1)
    archive.add(self.image2)
    assert len(archive) == 2
    assert archive.listing() == filenames
    archive.export()
    export_folder = os.fspath(tmp_path / export_name) + '.zip'
    assert os.path.isfile(export_folder)
    namelist = [f'{export_name}/{f}' for f in filenames]
    with zipfile.ZipFile(export_folder, 'r') as f:
        expected = sorted(map(os.path.abspath, namelist))
        output = sorted(map(os.path.abspath, f.namelist()))
        assert expected == output
    assert archive.listing() == []