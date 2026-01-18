from time import ctime, time
from zope.interface import implementer
from twisted import copyright
from twisted.cred import credentials, error as ecred, portal
from twisted.internet import defer, protocol
from twisted.python import failure, log, reflect
from twisted.python.components import registerAdapter
from twisted.spread import pb
from twisted.words import ewords, iwords
from twisted.words.protocols import irc
def irc_NAMES(self, prefix, params):
    """
        Names message

        Parameters: [ <channel> *( "," <channel> ) [ <target> ] ]
        """
    try:
        channel = params[-1]
        if isinstance(channel, bytes):
            channel = channel.decode(self.encoding)
    except UnicodeDecodeError:
        self.sendMessage(irc.ERR_NOSUCHCHANNEL, params[-1], ':No such channel (could not decode your unicode!)')
        return
    if channel.startswith('#'):
        channel = channel[1:]

    def cbGroup(group):
        self.names(self.name, '#' + group.name, [user.name for user in group.iterusers()])

    def ebGroup(err):
        err.trap(ewords.NoSuchGroup)
        self.names(self.name, '#' + channel, [])
    self.realm.lookupGroup(channel).addCallbacks(cbGroup, ebGroup)