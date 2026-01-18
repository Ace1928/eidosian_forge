import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def mlsd(self, path='', facts=[]):
    """List a directory in a standardized format by using MLSD
        command (RFC-3659). If path is omitted the current directory
        is assumed. "facts" is a list of strings representing the type
        of information desired (e.g. ["type", "size", "perm"]).

        Return a generator object yielding a tuple of two elements
        for every file found in path.
        First element is the file name, the second one is a dictionary
        including a variable number of "facts" depending on the server
        and whether "facts" argument has been provided.
        """
    if facts:
        self.sendcmd('OPTS MLST ' + ';'.join(facts) + ';')
    if path:
        cmd = 'MLSD %s' % path
    else:
        cmd = 'MLSD'
    lines = []
    self.retrlines(cmd, lines.append)
    for line in lines:
        facts_found, _, name = line.rstrip(CRLF).partition(' ')
        entry = {}
        for fact in facts_found[:-1].split(';'):
            key, _, value = fact.partition('=')
            entry[key.lower()] = value
        yield (name, entry)