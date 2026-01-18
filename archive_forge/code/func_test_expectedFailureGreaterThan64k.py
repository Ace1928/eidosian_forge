from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
def test_expectedFailureGreaterThan64k(self) -> None:
    """
        Fail, but expectedly.
        """
    raise RuntimeError('x' * (2 ** 16 + 1))