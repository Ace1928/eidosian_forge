from axolotl.state.prekeystore import PreKeyStore
from axolotl.state.prekeyrecord import PreKeyRecord
from yowsup.axolotl.exceptions import InvalidKeyIdException
import sys
def setAsSent(self, prekeyIds):
    """
        :param preKeyIds:
        :type preKeyIds: list
        :return:
        :rtype:
        """
    for prekeyId in prekeyIds:
        q = 'UPDATE prekeys SET sent_to_server = ? WHERE prekey_id = ?'
        cursor = self.dbConn.cursor()
        cursor.execute(q, (1, prekeyId))
    self.dbConn.commit()