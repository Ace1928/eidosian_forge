from struct import pack, unpack, calcsize, error, Struct
import os
import sys
import time
import array
import tempfile
import logging
import io
from datetime import date
import zipfile
def organize_polygon_rings(rings, return_errors=None):
    """Organize a list of coordinate rings into one or more polygons with holes.
    Returns a list of polygons, where each polygon is composed of a single exterior
    ring, and one or more interior holes. If a return_errors dict is provided (optional), 
    any errors encountered will be added to it. 

    Rings must be closed, and cannot intersect each other (non-self-intersecting polygon).
    Rings are determined as exteriors if they run in clockwise direction, or interior
    holes if they run in counter-clockwise direction. This method is used to construct
    GeoJSON (multi)polygons from the shapefile polygon shape type, which does not
    explicitly store the structure of the polygons beyond exterior/interior ring orientation. 
    """
    exteriors = []
    holes = []
    for ring in rings:
        if is_cw(ring):
            exteriors.append(ring)
        else:
            holes.append(ring)
    if len(exteriors) == 1:
        poly = [exteriors[0]] + holes
        polys = [poly]
        return polys
    elif len(exteriors) > 1:
        if not holes:
            polys = []
            for ext in exteriors:
                poly = [ext]
                polys.append(poly)
            return polys
        hole_exteriors = dict([(hole_i, []) for hole_i in xrange(len(holes))])
        exterior_bboxes = [ring_bbox(ring) for ring in exteriors]
        for hole_i in hole_exteriors.keys():
            hole_bbox = ring_bbox(holes[hole_i])
            for ext_i, ext_bbox in enumerate(exterior_bboxes):
                if bbox_contains(ext_bbox, hole_bbox):
                    hole_exteriors[hole_i].append(ext_i)
        for hole_i, exterior_candidates in hole_exteriors.items():
            if len(exterior_candidates) > 1:
                ccw = not is_cw(holes[hole_i])
                hole_sample = ring_sample(holes[hole_i], ccw=ccw)
                new_exterior_candidates = []
                for ext_i in exterior_candidates:
                    hole_in_exterior = ring_contains_point(exteriors[ext_i], hole_sample)
                    if hole_in_exterior:
                        new_exterior_candidates.append(ext_i)
                hole_exteriors[hole_i] = new_exterior_candidates
        for hole_i, exterior_candidates in hole_exteriors.items():
            if len(exterior_candidates) > 1:
                ext_i = sorted(exterior_candidates, key=lambda x: abs(signed_area(exteriors[x], fast=True)))[0]
                hole_exteriors[hole_i] = [ext_i]
        orphan_holes = []
        for hole_i, exterior_candidates in list(hole_exteriors.items()):
            if not exterior_candidates:
                orphan_holes.append(hole_i)
                del hole_exteriors[hole_i]
                continue
        polys = []
        for ext_i, ext in enumerate(exteriors):
            poly = [ext]
            poly_holes = []
            for hole_i, exterior_candidates in list(hole_exteriors.items()):
                if exterior_candidates[0] == ext_i:
                    poly_holes.append(holes[hole_i])
            poly += poly_holes
            polys.append(poly)
        for hole_i in orphan_holes:
            ext = holes[hole_i]
            poly = [ext]
            polys.append(poly)
        if orphan_holes and return_errors is not None:
            return_errors['polygon_orphaned_holes'] = len(orphan_holes)
        return polys
    else:
        if return_errors is not None:
            return_errors['polygon_only_holes'] = len(holes)
        exteriors = holes
        polys = [[ext] for ext in exteriors]
        return polys