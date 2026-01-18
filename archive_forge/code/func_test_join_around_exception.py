import threading
from .. import cethread, tests
def test_join_around_exception(self):
    resume = threading.Event()

    class MyException(Exception):
        pass

    def raise_my_exception():
        resume.wait()
        raise MyException()
    tt = cethread.CatchingExceptionThread(target=raise_my_exception)
    tt.start()
    tt.join(timeout=0)
    self.assertIs(None, tt.exception)
    resume.set()
    self.assertRaises(MyException, tt.join)