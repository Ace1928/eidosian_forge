from pyxnat import uriutil
def test_join_uri():
    uri = uriutil.join_uri('/projects', 'project_id', 'subjects', 'subject_id')
    assert uri == '/projects/project_id/subjects/subject_id'