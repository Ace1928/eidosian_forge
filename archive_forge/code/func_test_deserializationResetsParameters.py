import pickle
from twisted.internet.protocol import Protocol, ReconnectingClientFactory
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_deserializationResetsParameters(self):
    """
        A L{ReconnectingClientFactory} which is unpickled does not have an
        L{IConnector} and has its reconnecting timing parameters reset to their
        initial values.
        """
    factory = ReconnectingClientFactory()
    factory.clientConnectionFailed(FakeConnector(), None)
    self.addCleanup(factory.stopTrying)
    serialized = pickle.dumps(factory)
    unserialized = pickle.loads(serialized)
    self.assertIsNone(unserialized.connector)
    self.assertIsNone(unserialized._callID)
    self.assertEqual(unserialized.retries, 0)
    self.assertEqual(unserialized.delay, factory.initialDelay)
    self.assertTrue(unserialized.continueTrying)