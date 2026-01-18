import logging
import unittest
import os
import gensim.downloader as api
import shutil
import numpy as np
def test_load_model(self):
    if os.path.isdir(api.BASE_DIR):
        shutil.rmtree(api.BASE_DIR)
    vector_dead = np.array([0.17403787, -0.10167074, -0.00950371, -0.10367849, -0.14034484, -0.08751217, 0.10030612, 0.07677923, -0.32563496, 0.01929072, 0.20521086, -0.1617067, 0.00475458, 0.21956187, -0.08783089, -0.05937332, 0.26528183, -0.06771874, -0.12369668, 0.12020949, 0.28731, 0.36735833, 0.28051138, -0.10407482, 0.2496888, -0.19372769, -0.28719661, 0.11989869, -0.00393865, -0.2431484, 0.02725661, -0.20421691, 0.0328669, -0.26947051, -0.08068217, -0.10245913, 0.1170633, 0.16583319, 0.1183883, -0.11217165, 0.1261425, -0.0319365, -0.15787181, 0.03753783, 0.14748634, 0.00414471, -0.02296237, 0.18336892, -0.23840059, 0.17924534])
    dataset_path = os.path.join(api.BASE_DIR, '__testing_word2vec-matrix-synopsis', '__testing_word2vec-matrix-synopsis.gz')
    model = api.load('__testing_word2vec-matrix-synopsis')
    vector_dead_calc = model.wv['dead']
    self.assertTrue(np.allclose(vector_dead, vector_dead_calc))
    shutil.rmtree(api.BASE_DIR)
    self.assertEqual(api.load('__testing_word2vec-matrix-synopsis', return_path=True), dataset_path)
    shutil.rmtree(api.BASE_DIR)