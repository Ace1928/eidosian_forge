from twisted.internet import threads
from twisted.python import log, reflect
def runQuery(self, *args, **kw):
    """
        Execute an SQL query and return the result.

        A DB-API cursor which will be invoked with C{cursor.execute(*args,
        **kw)}. The exact nature of the arguments will depend on the specific
        flavor of DB-API being used, but the first argument in C{*args} be an
        SQL statement. The result of a subsequent C{cursor.fetchall()} will be
        fired to the L{Deferred} which is returned. If either the 'execute' or
        'fetchall' methods raise an exception, the transaction will be rolled
        back and a L{twisted.python.failure.Failure} returned.

        The C{*args} and C{**kw} arguments will be passed to the DB-API
        cursor's 'execute' method.

        @return: a L{Deferred} which will fire the return value of a DB-API
            cursor's 'fetchall' method, or a L{twisted.python.failure.Failure}.
        """
    return self.runInteraction(self._runQuery, *args, **kw)