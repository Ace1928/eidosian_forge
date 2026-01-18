from breezy import controldir, transport
from breezy.tests import TestNotApplicable
from breezy.tests.per_repository import TestCaseWithRepository
def assertSameRepo(self, a, b):
    """Asserts that two objects are the same repository.

        This method does the comparison both ways (`a.has_same_location(b)` as
        well as `b.has_same_location(a)`) to make sure both objects'
        `has_same_location` methods give the same results.
        """
    self.assertTrue(a.has_same_location(b), '{!r} is not the same repository as {!r}'.format(a, b))
    self.assertTrue(b.has_same_location(a), '{!r} is the same as {!r}, but not vice versa'.format(a, b))