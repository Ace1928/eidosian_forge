import os
import unittest
from xml.dom.minidom import parseString as parse_xml_string
from shapely.geometry import (
from shapely.geometry.collection import GeometryCollection
def assertSVG(self, geom, expected, **kwrds):
    """Helper function to check XML and debug SVG"""
    svg_elem = geom.svg(**kwrds)
    try:
        parse_xml_string(svg_elem)
    except Exception:
        raise AssertionError('XML is not valid for SVG element: ' + str(svg_elem))
    svg_doc = geom._repr_svg_()
    try:
        doc = parse_xml_string(svg_doc)
    except Exception:
        raise AssertionError('XML is not valid for SVG document: ' + str(svg_doc))
    svg_output_dir = None
    if svg_output_dir:
        fname = geom.geom_type
        if geom.is_empty:
            fname += '_empty'
        if not geom.is_valid:
            fname += '_invalid'
        if kwrds:
            fname += '_' + ','.join((str(k) + '=' + str(kwrds[k]) for k in kwrds))
        svg_path = os.path.join(svg_output_dir, fname + '.svg')
        with open(svg_path, 'w') as fp:
            fp.write(doc.toprettyxml())
    assert svg_elem == expected