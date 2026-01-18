def test_nbextension_path():
    from ipydatawidgets import _jupyter_nbextension_paths
    path = _jupyter_nbextension_paths()
    assert len(path) == 1
    assert isinstance(path[0], dict)