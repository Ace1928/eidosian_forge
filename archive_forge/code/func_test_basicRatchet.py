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
def test_basicRatchet(self):
    aliceStore = InMemorySenderKeyStore()
    bobStore = InMemorySenderKeyStore()
    aliceSessionBuilder = GroupSessionBuilder(aliceStore)
    bobSessionBuilder = GroupSessionBuilder(bobStore)
    aliceGroupCipher = GroupCipher(aliceStore, 'groupWithBobInIt')
    bobGroupCipher = GroupCipher(bobStore, 'groupWithBobInIt::aliceUserName')
    aliceGroupCipher = GroupCipher(aliceStore, GROUP_SENDER)
    bobGroupCipher = GroupCipher(bobStore, GROUP_SENDER)
    sentAliceDistributionMessage = aliceSessionBuilder.create(GROUP_SENDER)
    receivedAliceDistributionMessage = SenderKeyDistributionMessage(serialized=sentAliceDistributionMessage.serialize())
    bobSessionBuilder.process(GROUP_SENDER, receivedAliceDistributionMessage)
    ciphertextFromAlice = aliceGroupCipher.encrypt(b'smert ze smert')
    ciphertextFromAlice2 = aliceGroupCipher.encrypt(b'smert ze smert2')
    ciphertextFromAlice3 = aliceGroupCipher.encrypt(b'smert ze smert3')
    plaintextFromAlice = bobGroupCipher.decrypt(ciphertextFromAlice)
    try:
        bobGroupCipher.decrypt(ciphertextFromAlice)
        raise AssertionError('Should have ratcheted forward!')
    except DuplicateMessageException as dme:
        pass
    plaintextFromAlice2 = bobGroupCipher.decrypt(ciphertextFromAlice2)
    plaintextFromAlice3 = bobGroupCipher.decrypt(ciphertextFromAlice3)
    self.assertEqual(plaintextFromAlice, b'smert ze smert')
    self.assertEqual(plaintextFromAlice2, b'smert ze smert2')
    self.assertEqual(plaintextFromAlice3, b'smert ze smert3')