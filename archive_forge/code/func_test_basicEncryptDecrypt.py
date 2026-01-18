import unittest
from .inmemorysenderkeystore import InMemorySenderKeyStore
from ...groups.groupsessionbuilder import GroupSessionBuilder
from ...util.keyhelper import KeyHelper
from ...groups.groupcipher import GroupCipher
from ...duplicatemessagexception import DuplicateMessageException
from ...nosessionexception import NoSessionException
from ...groups.senderkeyname import SenderKeyName
from ...axolotladdress import AxolotlAddress
from ...protocol.senderkeydistributionmessage import SenderKeyDistributionMessage
def test_basicEncryptDecrypt(self):
    aliceStore = InMemorySenderKeyStore()
    bobStore = InMemorySenderKeyStore()
    aliceSessionBuilder = GroupSessionBuilder(aliceStore)
    bobSessionBuilder = GroupSessionBuilder(bobStore)
    aliceGroupCipher = GroupCipher(aliceStore, GROUP_SENDER)
    bobGroupCipher = GroupCipher(bobStore, GROUP_SENDER)
    sentAliceDistributionMessage = aliceSessionBuilder.create(GROUP_SENDER)
    receivedAliceDistributionMessage = SenderKeyDistributionMessage(serialized=sentAliceDistributionMessage.serialize())
    bobSessionBuilder.process(GROUP_SENDER, receivedAliceDistributionMessage)
    ciphertextFromAlice = aliceGroupCipher.encrypt(b'smert ze smert')
    plaintextFromAlice = bobGroupCipher.decrypt(ciphertextFromAlice)
    self.assertEqual(plaintextFromAlice, b'smert ze smert')