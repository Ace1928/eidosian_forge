from pathlib import Path
from IPython.utils.tempdir import NamedFileInTemporaryDirectory
from IPython.utils.tempdir import TemporaryWorkingDirectory
def test_named_file_in_temporary_directory():
    with NamedFileInTemporaryDirectory('filename') as file:
        name = file.name
        assert not file.closed
        assert Path(name).exists()
        file.write(b'test')
    assert file.closed
    assert not Path(name).exists()