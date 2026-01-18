import json
import os
import tarfile
import zipfile
import numpy as np
from holoviews import Image
from holoviews.core.io import FileArchive, Serializer
def test_filearchive_clear_file(self, tmp_path):
    export_name = 'archive_for_clear'
    export_name = 'archive_for_clear'
    archive = FileArchive(root=os.fspath(tmp_path), export_name=export_name, pack=False)
    archive.add(self.image1)
    archive.add(self.image2)
    archive.clear()
    assert archive._files == {}