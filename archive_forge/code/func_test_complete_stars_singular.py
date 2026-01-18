from pyxnat import select
def test_complete_stars_singular():
    assert select.compute('/project/subject/experiment') == ['/projects/*/subjects/*/experiments/*']