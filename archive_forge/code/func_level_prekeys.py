from axolotl.util.keyhelper import KeyHelper
from axolotl.identitykeypair import IdentityKeyPair
from axolotl.groups.senderkeyname import SenderKeyName
from axolotl.axolotladdress import AxolotlAddress
from axolotl.sessioncipher import SessionCipher
from axolotl.groups.groupcipher import GroupCipher
from axolotl.groups.groupsessionbuilder import GroupSessionBuilder
from axolotl.sessionbuilder import SessionBuilder
from axolotl.protocol.prekeywhispermessage import PreKeyWhisperMessage
from axolotl.protocol.whispermessage import WhisperMessage
from axolotl.state.prekeybundle import PreKeyBundle
from axolotl.untrustedidentityexception import UntrustedIdentityException
from axolotl.invalidmessageexception import InvalidMessageException
from axolotl.duplicatemessagexception import DuplicateMessageException
from axolotl.invalidkeyidexception import InvalidKeyIdException
from axolotl.nosessionexception import NoSessionException
from axolotl.protocol.senderkeydistributionmessage import SenderKeyDistributionMessage
from axolotl.state.axolotlstore import AxolotlStore
from yowsup.axolotl.store.sqlite.liteaxolotlstore import LiteAxolotlStore
from yowsup.axolotl import exceptions
import random
import logging
import sys
def level_prekeys(self, force=False):
    logger.debug('level_prekeys(force=%s)' % force)
    len_pending_prekeys = len(self._store.loadPreKeys())
    logger.debug('len(pending_prekeys) = %d' % len_pending_prekeys)
    if force or len_pending_prekeys < self.THRESHOLD_REGEN:
        count_gen = self.COUNT_GEN_PREKEYS
        max_prekey_id = self._store.preKeyStore.loadMaxPreKeyId()
        logger.info('Generating %d prekeys, current max_prekey_id=%d' % (count_gen, max_prekey_id))
        prekeys = KeyHelper.generatePreKeys(max_prekey_id + 1, count_gen)
        logger.info('Storing %d prekeys' % len(prekeys))
        for i in range(0, len(prekeys)):
            key = prekeys[i]
            if logger.level <= logging.DEBUG:
                sys.stdout.write('Storing prekey %d/%d \r' % (i + 1, len(prekeys)))
                sys.stdout.flush()
            self._store.storePreKey(key.getId(), key)
        return prekeys
    return []