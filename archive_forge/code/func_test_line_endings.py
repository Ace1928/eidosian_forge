import os
from os.path import abspath, dirname
def test_line_endings():
    rootdir = dirname(dirname(abspath(__file__)))
    for subdir, dirs, files in os.walk(rootdir):
        if any((i in subdir for i in ['.git', '.idea', '__pycache__'])):
            continue
        for file in files:
            if file.endswith('.parquet'):
                continue
            filepath = os.path.join(subdir, file)
            with open(filepath, 'rb+') as f:
                file_contents = f.read()
                new_contents = file_contents.replace(b'\r\n', b'\n')
                assert new_contents == file_contents, 'File has CRLF: {}'.format(filepath)