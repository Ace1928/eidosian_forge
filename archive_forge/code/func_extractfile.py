from pathlib import Path
import sys
def extractfile(self, member):
    """ Return a file object corresponding to the requested member. A member
        can be specified either as a string (its name) or as a ArMember
        instance. """
    for m in self.__members:
        if isinstance(member, ArMember) and m.name == member.name:
            return m
        if member == m.name:
            return m
    return None