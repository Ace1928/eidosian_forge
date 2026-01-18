import re
from nibabel.optpkg import optional_package
from ..utils import find_private_section as fps
from .test_dicomwrappers import DATA, DATA_PHILIPS
def test_find_private_section_fake():
    ds = pydicom.dataset.Dataset({})
    assert fps(ds, 17, 'some section') is None
    ds.add_new((17, 16), 'LO', b'some section')
    assert fps(ds, 17, 'some section') == 4096
    ds.add_new((17, 17), 'LO', b'another section')
    ds.add_new((17, 18), 'LO', b'third section')
    assert fps(ds, 17, 'third section') == 4608
    ds.add_new((17, 18), 'OB', b'third section')
    assert fps(ds, 17, 'third section') == 4608
    ds.add_new((17, 18), 'PN', b'third section')
    assert fps(ds, 17, 'third section') is None
    ds.add_new((17, 18), 'LO', 'third section')
    assert fps(ds, 17, 'third section') == 4608
    ds.add_new((17, 18), 'LO', b'third section')
    assert fps(ds, 17, b'third section') == 4608
    assert fps(ds, 17, b'third sectio') is None
    assert fps(ds, 17, 'hird sectio') is None
    assert fps(ds, 17, re.compile('third\\Wsectio[nN]')) == 4608
    assert fps(ds, 17, re.compile('not third\\Wsectio[nN]')) is None
    ds.add_new((17, 19), 'LO', b'near section')
    assert fps(ds, 17, 'near section') == 4864
    ds.add_new((17, 21), 'LO', b'far section')
    assert fps(ds, 17, 'far section') == 5376
    assert fps(ds, 17, re.compile('(another|third) section')) == 4352
    ds = pydicom.dataset.Dataset({})
    ds.add_new((17, 255), 'LO', b'some section')
    assert fps(ds, 17, 'some section') == 65280
    ds = pydicom.dataset.Dataset({})
    ds.add_new((17, 256), 'LO', b'some section')
    assert fps(ds, 17, 'some section') is None