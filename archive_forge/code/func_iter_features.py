from itertools import chain
import json
import re
import click
def iter_features(geojsonfile, func=None):
    """Extract GeoJSON features from a text file object.

    Given a file-like object containing a single GeoJSON feature
    collection text or a sequence of GeoJSON features, iter_features()
    iterates over lines of the file and yields GeoJSON features.

    Parameters
    ----------
    geojsonfile: a file-like object
        The geojsonfile implements the iterator protocol and yields
        lines of JSON text.
    func: function, optional
        A function that will be applied to each extracted feature. It
        takes a feature object and may return a replacement feature or
        None -- in which case iter_features does not yield.

    Yields
    ------
    Mapping
        A GeoJSON Feature represented by a Python mapping

    """
    func = func or (lambda x: x)
    first_line = next(geojsonfile)
    if first_line.startswith(u'\x1e'):
        text_buffer = first_line.strip(u'\x1e')
        for line in geojsonfile:
            if line.startswith(u'\x1e'):
                if text_buffer:
                    obj = json.loads(text_buffer)
                    if 'coordinates' in obj:
                        obj = to_feature(obj)
                    newfeat = func(obj)
                    if newfeat:
                        yield newfeat
                text_buffer = line.strip(u'\x1e')
            else:
                text_buffer += line
        else:
            obj = json.loads(text_buffer)
            if 'coordinates' in obj:
                obj = to_feature(obj)
            newfeat = func(obj)
            if newfeat:
                yield newfeat
    else:
        try:
            obj = json.loads(first_line)
            if obj['type'] == 'Feature':
                newfeat = func(obj)
                if newfeat:
                    yield newfeat
                for line in geojsonfile:
                    newfeat = func(json.loads(line))
                    if newfeat:
                        yield newfeat
            elif obj['type'] == 'FeatureCollection':
                for feat in obj['features']:
                    newfeat = func(feat)
                    if newfeat:
                        yield newfeat
            elif 'coordinates' in obj:
                newfeat = func(to_feature(obj))
                if newfeat:
                    yield newfeat
                for line in geojsonfile:
                    newfeat = func(to_feature(json.loads(line)))
                    if newfeat:
                        yield newfeat
        except ValueError:
            text = ''.join(chain([first_line], geojsonfile))
            obj = json.loads(text)
            if obj['type'] == 'Feature':
                newfeat = func(obj)
                if newfeat:
                    yield newfeat
            elif obj['type'] == 'FeatureCollection':
                for feat in obj['features']:
                    newfeat = func(feat)
                    if newfeat:
                        yield newfeat
            elif 'coordinates' in obj:
                newfeat = func(to_feature(obj))
                if newfeat:
                    yield newfeat