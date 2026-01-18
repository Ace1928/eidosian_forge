import threading
from .. import cethread, tests
def test_exception_is_re_raised(self):

    class MyException(Exception):
        pass

    def raise_my_exception():
        raise MyException()
    tt = cethread.CatchingExceptionThread(target=raise_my_exception)
    tt.start()
    self.assertRaises(MyException, tt.join)