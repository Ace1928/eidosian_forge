from .pysqlite import SQLiteDialect_pysqlite
from ... import pool
def on_connect_url(self, url):
    super_on_connect = super().on_connect_url(url)
    passphrase = url.password or ''
    url_query = dict(url.query)

    def on_connect(conn):
        cursor = conn.cursor()
        cursor.execute('pragma key="%s"' % passphrase)
        for prag in self.pragmas:
            value = url_query.get(prag, None)
            if value is not None:
                cursor.execute('pragma %s="%s"' % (prag, value))
        cursor.close()
        if super_on_connect:
            super_on_connect(conn)
    return on_connect