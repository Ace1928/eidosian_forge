from __future__ import division
import logging
import click
import cligj
import rasterio
from rasterio.rio import options
from rasterio.features import dataset_features
from rasterio.rio.helpers import write_features
def feature_gen(src, env, *args, **kwargs):

    class Collection:

        def __init__(self, env):
            self.bboxes = []
            self.env = env

        @property
        def bbox(self):
            minxs, minys, maxxs, maxys = zip(*self.bboxes)
            return (min(minxs), min(minys), max(maxxs), max(maxys))

        def __call__(self):
            for f in dataset_features(src, *args, **kwargs):
                self.bboxes.append(f['bbox'])
                yield f
    return Collection(env)