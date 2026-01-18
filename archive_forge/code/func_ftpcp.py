import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def ftpcp(source, sourcename, target, targetname='', type='I'):
    """Copy file from one FTP-instance to another."""
    if not targetname:
        targetname = sourcename
    type = 'TYPE ' + type
    source.voidcmd(type)
    target.voidcmd(type)
    sourcehost, sourceport = parse227(source.sendcmd('PASV'))
    target.sendport(sourcehost, sourceport)
    treply = target.sendcmd('STOR ' + targetname)
    if treply[:3] not in {'125', '150'}:
        raise error_proto
    sreply = source.sendcmd('RETR ' + sourcename)
    if sreply[:3] not in {'125', '150'}:
        raise error_proto
    source.voidresp()
    target.voidresp()