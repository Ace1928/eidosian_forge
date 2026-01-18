import warnings
from twisted.trial.unittest import TestCase
def test_asForeignClassAttributeViaInstance(self):
    """
        A constant defined on a L{Names} subclass may be set as an attribute of
        another class and then retrieved from an instance of that class using
        that attribute.
        """

    class Another:
        something = self.METHOD.GET
    self.assertIs(self.METHOD.GET, Another().something)