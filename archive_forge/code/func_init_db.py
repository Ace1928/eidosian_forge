from PySide2.QtSql import QSqlDatabase, QSqlError, QSqlQuery
from datetime import date
def init_db():
    """
    init_db()
    Initializes the database.
    If tables "books" and "authors" are already in the database, do nothing.
    Return value: None or raises ValueError
    The error value is the QtSql error instance.
    """

    def check(func, *args):
        if not func(*args):
            raise ValueError(func.__self__.lastError())
    db = QSqlDatabase.addDatabase('QSQLITE')
    db.setDatabaseName(':memory:')
    check(db.open)
    q = QSqlQuery()
    check(q.exec_, BOOKS_SQL)
    check(q.exec_, AUTHORS_SQL)
    check(q.exec_, GENRES_SQL)
    check(q.prepare, INSERT_AUTHOR_SQL)
    asimovId = add_author(q, 'Isaac Asimov', date(1920, 2, 1))
    greeneId = add_author(q, 'Graham Greene', date(1904, 10, 2))
    pratchettId = add_author(q, 'Terry Pratchett', date(1948, 4, 28))
    check(q.prepare, INSERT_GENRE_SQL)
    sfiction = add_genre(q, 'Science Fiction')
    fiction = add_genre(q, 'Fiction')
    fantasy = add_genre(q, 'Fantasy')
    check(q.prepare, INSERT_BOOK_SQL)
    add_book(q, 'Foundation', 1951, asimovId, sfiction, 3)
    add_book(q, 'Foundation and Empire', 1952, asimovId, sfiction, 4)
    add_book(q, 'Second Foundation', 1953, asimovId, sfiction, 3)
    add_book(q, "Foundation's Edge", 1982, asimovId, sfiction, 3)
    add_book(q, 'Foundation and Earth', 1986, asimovId, sfiction, 4)
    add_book(q, 'Prelude to Foundation', 1988, asimovId, sfiction, 3)
    add_book(q, 'Forward the Foundation', 1993, asimovId, sfiction, 3)
    add_book(q, 'The Power and the Glory', 1940, greeneId, fiction, 4)
    add_book(q, 'The Third Man', 1950, greeneId, fiction, 5)
    add_book(q, 'Our Man in Havana', 1958, greeneId, fiction, 4)
    add_book(q, 'Guards! Guards!', 1989, pratchettId, fantasy, 3)
    add_book(q, 'Night Watch', 2002, pratchettId, fantasy, 3)
    add_book(q, 'Going Postal', 2004, pratchettId, fantasy, 3)