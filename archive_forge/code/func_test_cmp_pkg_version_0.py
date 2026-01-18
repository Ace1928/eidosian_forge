import pytest
import nibabel as nib
from nibabel.pkg_info import cmp_pkg_version
def test_cmp_pkg_version_0():
    assert cmp_pkg_version(nib.__version__) == 0
    assert cmp_pkg_version('0.0') == -1
    assert cmp_pkg_version('1000.1000.1') == 1
    assert cmp_pkg_version(nib.__version__, nib.__version__) == 0
    seq = ('3.0.0dev', '3.0.0rc1', '3.0.0rc1.post.dev', '3.0.0rc2', '3.0.0rc2.post.dev', '3.0.0')
    for stage1, stage2 in zip(seq[:-1], seq[1:]):
        assert cmp_pkg_version(stage1, stage2) == -1
        assert cmp_pkg_version(stage2, stage1) == 1