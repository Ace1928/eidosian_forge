import tempfile
import os
from gitdb.test.lib import TestBase
from gitdb.util import (
def test_lockedfd(self):
    my_file = tempfile.mktemp()
    orig_data = 'hello'
    new_data = 'world'
    with open(my_file, 'wb') as my_file_fp:
        my_file_fp.write(orig_data.encode('ascii'))
    try:
        lfd = LockedFD(my_file)
        lockfilepath = lfd._lockfilepath()
        self.assertRaises(AssertionError, lfd.rollback)
        self.assertRaises(AssertionError, lfd.commit)
        assert not os.path.isfile(lockfilepath)
        wfd = lfd.open(write=True)
        assert lfd._fd is wfd
        assert os.path.isfile(lockfilepath)
        os.write(wfd, new_data.encode('ascii'))
        lfd.rollback()
        assert lfd._fd is None
        self._cmp_contents(my_file, orig_data)
        assert not os.path.isfile(lockfilepath)
        lfd.commit()
        lfd.rollback()
        lfd = LockedFD(my_file)
        rfd = lfd.open(write=False)
        assert os.read(rfd, len(orig_data)) == orig_data.encode('ascii')
        assert os.path.isfile(lockfilepath)
        del lfd
        assert not os.path.isfile(lockfilepath)
        lfd = LockedFD(my_file)
        olfd = LockedFD(my_file)
        assert not os.path.isfile(lockfilepath)
        wfdstream = lfd.open(write=True, stream=True)
        assert os.path.isfile(lockfilepath)
        self.assertRaises(IOError, olfd.open)
        wfdstream.write(new_data.encode('ascii'))
        lfd.commit()
        assert not os.path.isfile(lockfilepath)
        self._cmp_contents(my_file, new_data)
    finally:
        os.remove(my_file)
    lfd = LockedFD(tempfile.mktemp())
    try:
        lfd.open(write=False)
    except OSError:
        assert not os.path.exists(lfd._lockfilepath())
    else:
        self.fail('expected OSError')