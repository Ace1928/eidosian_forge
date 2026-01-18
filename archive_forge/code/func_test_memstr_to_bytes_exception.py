from __future__ import with_statement
import array
import os
from joblib.disk import disk_used, memstr_to_bytes, mkdirp, rm_subdirs
from joblib.testing import parametrize, raises
@parametrize('text,exception,regex', [('fooG', ValueError, 'Invalid literal for size.*fooG.*'), ('1.4N', ValueError, 'Invalid literal for size.*1.4N.*')])
def test_memstr_to_bytes_exception(text, exception, regex):
    with raises(exception) as excinfo:
        memstr_to_bytes(text)
    assert excinfo.match(regex)