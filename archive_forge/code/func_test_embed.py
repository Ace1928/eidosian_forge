from __future__ import absolute_import
import os
import shutil
import json
import contextlib
import numpy as np
import pytest
import ipyvolume
import ipyvolume.pylab as p3
import ipyvolume as ipv
import ipyvolume.examples
import ipyvolume.datasets
import ipyvolume.utils
import ipyvolume.serialize
def test_embed():
    p3.clear()
    x, y, z = np.random.random((3, 100))
    p3.scatter(x, y, z)
    p3.save('tmp/ipyolume_scatter_online.html', offline=False, devmode=True)
    assert os.path.getsize('tmp/ipyolume_scatter_online.html') > 0
    p3.save('tmp/ipyolume_scatter_offline.html', offline=True, scripts_path='js/subdir', devmode=True)
    assert os.path.getsize('tmp/ipyolume_scatter_offline.html') > 0