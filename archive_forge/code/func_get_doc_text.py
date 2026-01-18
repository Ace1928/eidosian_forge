import sqlite3
from . import utils
def get_doc_text(self, doc_id):
    """
        Fetch the raw text of the doc for 'doc_id'.
        """
    cursor = self.connection.cursor()
    cursor.execute('SELECT text FROM documents WHERE id = ?', (utils.normalize(doc_id),))
    result = cursor.fetchone()
    cursor.close()
    return result if result is None else result[0]