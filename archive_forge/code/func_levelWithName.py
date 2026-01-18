from constantly import NamedConstant, Names
@classmethod
def levelWithName(cls, name: str) -> NamedConstant:
    """
        Get the log level with the given name.

        @param name: The name of a log level.

        @return: The L{LogLevel} with the specified C{name}.

        @raise InvalidLogLevelError: if the C{name} does not name a valid log
            level.
        """
    try:
        return cls.lookupByName(name)
    except ValueError:
        raise InvalidLogLevelError(name)