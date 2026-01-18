from axolotl.state.prekeystore import PreKeyStore
from axolotl.state.prekeyrecord import PreKeyRecord
from yowsup.axolotl.exceptions import InvalidKeyIdException
import sys
def loadUnsentPendingPreKeys(self):
    q = 'SELECT record FROM prekeys WHERE sent_to_server is NULL or sent_to_server = ?'
    cursor = self.dbConn.cursor()
    cursor.execute(q, (0,))
    result = cursor.fetchall()
    return [PreKeyRecord(serialized=result[0]) for result in result]