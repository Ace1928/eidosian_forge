import json
import os
import tarfile
import zipfile
import numpy as np
from holoviews import Image
from holoviews.core.io import FileArchive, Serializer
def test_filearchive_image_pickle_tar(self, tmp_path):
    export_name = 'archive_image'
    filenames = ['Group1-Im1.hvz', 'Group2-Im2.hvz']
    archive = FileArchive(root=os.fspath(tmp_path), export_name=export_name, pack=True, archive_format='tar')
    archive.add(self.image1)
    archive.add(self.image2)
    assert len(archive) == 2
    assert archive.listing() == filenames
    archive.export()
    export_folder = os.fspath(tmp_path / export_name) + '.tar'
    assert os.path.isfile(export_folder)
    namelist = [f'{export_name}/{f}' for f in filenames]
    with tarfile.TarFile(export_folder, 'r') as f:
        assert sorted(namelist) == sorted([el.path for el in f.getmembers()])
    assert archive.listing() == []