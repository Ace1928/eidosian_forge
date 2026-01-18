import json
from pathlib import Path
import shutil
import matplotlib.dviread as dr
import pytest
@pytest.mark.skipif(shutil.which('kpsewhich') is None, reason='kpsewhich is not available')
def test_dviread():
    dirpath = Path(__file__).parent / 'baseline_images/dviread'
    with (dirpath / 'test.json').open() as f:
        correct = json.load(f)
    with dr.Dvi(str(dirpath / 'test.dvi'), None) as dvi:
        data = [{'text': [[t.x, t.y, chr(t.glyph), t.font.texname.decode('ascii'), round(t.font.size, 2)] for t in page.text], 'boxes': [[b.x, b.y, b.height, b.width] for b in page.boxes]} for page in dvi]
    assert data == correct