import sqlite3
from . import utils
def get_doc_ids(self):
    """
        Fetch all ids of docs stored in the db.
        """
    cursor = self.connection.cursor()
    cursor.execute('SELECT id FROM documents')
    results = [r[0] for r in cursor.fetchall()]
    cursor.close()
    return results