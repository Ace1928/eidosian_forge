from pyxnat import select
def test_complete_stars_plural():
    assert select.compute('/projects/subjects/experiments') == ['/projects/*/subjects/*/experiments/*']