import logging
import unittest
import os
import gensim.downloader as api
import shutil
import numpy as np
def test_multipart_load(self):
    dataset_path = os.path.join(api.BASE_DIR, '__testing_multipart-matrix-synopsis', '__testing_multipart-matrix-synopsis.gz')
    if os.path.isdir(api.BASE_DIR):
        shutil.rmtree(api.BASE_DIR)
    self.assertEqual(dataset_path, api.load('__testing_multipart-matrix-synopsis', return_path=True))
    shutil.rmtree(api.BASE_DIR)
    dataset = api.load('__testing_multipart-matrix-synopsis')
    self.assertEqual(len(list(dataset)), 1)