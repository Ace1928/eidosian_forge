import numpy as np
import pytest
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory
from ...testing import (
def create_label_map(applies_to_matrix_dimension):
    maps = []
    for name, meta, label in labels:
        label_table = ci.Cifti2LabelTable()
        for key, (tag, rgba) in label.items():
            label_table[key] = ci.Cifti2Label(key, tag, *rgba)
        maps.append(ci.Cifti2NamedMap(name, ci.Cifti2MetaData(meta), label_table))
    return ci.Cifti2MatrixIndicesMap(applies_to_matrix_dimension, 'CIFTI_INDEX_TYPE_LABELS', maps=maps)