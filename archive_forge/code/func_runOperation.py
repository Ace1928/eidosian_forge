from twisted.internet import threads
from twisted.python import log, reflect
def runOperation(self, *args, **kw):
    """
        Execute an SQL query and return L{None}.

        A DB-API cursor which will be invoked with C{cursor.execute(*args,
        **kw)}. The exact nature of the arguments will depend on the specific
        flavor of DB-API being used, but the first argument in C{*args} will be
        an SQL statement. This method will not attempt to fetch any results
        from the query and is thus suitable for C{INSERT}, C{DELETE}, and other
        SQL statements which do not return values. If the 'execute' method
        raises an exception, the transaction will be rolled back and a
        L{Failure} returned.

        The C{*args} and C{*kw} arguments will be passed to the DB-API cursor's
        'execute' method.

        @return: a L{Deferred} which will fire with L{None} or a
            L{twisted.python.failure.Failure}.
        """
    return self.runInteraction(self._runOperation, *args, **kw)