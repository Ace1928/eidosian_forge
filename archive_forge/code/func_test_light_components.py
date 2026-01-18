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
def test_light_components():
    ambient = ipyvolume.light_ambient()
    assert ambient.type == 'AmbientLight'
    assert ambient.color == 'white'
    assert ambient.intensity == 1
    hemisphere = ipyvolume.light_hemisphere()
    assert hemisphere.type == 'HemisphereLight'
    assert hemisphere.color == '#ffffbb'
    assert hemisphere.groundColor == '#080820'
    directional = ipyvolume.light_directional()
    assert directional.color == 'white'
    spot = ipyvolume.light_spot()
    assert spot.color == 'white'
    point = ipyvolume.light_point()
    assert point.color == 'white'