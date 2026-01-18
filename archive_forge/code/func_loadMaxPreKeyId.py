from axolotl.state.prekeystore import PreKeyStore
from axolotl.state.prekeyrecord import PreKeyRecord
from yowsup.axolotl.exceptions import InvalidKeyIdException
import sys
def loadMaxPreKeyId(self):
    q = 'SELECT max(prekey_id) FROM prekeys'
    cursor = self.dbConn.cursor()
    cursor.execute(q)
    result = cursor.fetchone()
    return 0 if result[0] is None else result[0]