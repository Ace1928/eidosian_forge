from .. import imageglobals as igs
def test_errorlevel():
    orig_level = igs.error_level
    for level in (10, 20, 30):
        with igs.ErrorLevel(level):
            assert igs.error_level == level
        assert igs.error_level == orig_level