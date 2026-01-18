import warnings
from itertools import product
import numpy as np
import pytest
from ..filebasedimages import FileBasedHeader, FileBasedImage, SerializableImage
from .test_image_api import GenericImageAPI, SerializeMixin
def test_filebased_header():

    class H(FileBasedHeader):

        def __init__(self, seq=None):
            if seq is None:
                seq = []
            self.a_list = list(seq)
    in_list = [1, 3, 2]
    hdr = H(in_list)
    hdr_c = hdr.copy()
    assert hdr_c.a_list == hdr.a_list
    hdr_c.a_list[0] = 99
    assert hdr_c.a_list != hdr.a_list
    hdr2 = H.from_header(hdr)
    assert isinstance(hdr2, H)
    assert hdr2.a_list == hdr.a_list
    hdr2.a_list[0] = 42
    assert hdr2.a_list != hdr.a_list
    hdr3 = H.from_header()
    assert isinstance(hdr3, H)
    assert hdr3.a_list == []
    hdr4 = H.from_header(None)
    assert isinstance(hdr4, H)
    assert hdr4.a_list == []