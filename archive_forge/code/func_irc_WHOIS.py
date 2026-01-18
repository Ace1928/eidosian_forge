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
def irc_WHOIS(self, prefix, params):
    """
        Whois query

        Parameters: [ <target> ] <mask> *( "," <mask> )
        """

    def cbUser(user):
        self.whois(self.name, user.name, user.name, self.realm.name, user.name, self.realm.name, 'Hi mom!', False, int(time() - user.lastMessage), user.signOn, ['#' + group.name for group in user.itergroups()])

    def ebUser(err):
        err.trap(ewords.NoSuchUser)
        self.sendMessage(irc.ERR_NOSUCHNICK, params[0], ':No such nick/channel')
    try:
        user = params[0]
        if isinstance(user, bytes):
            user = user.decode(self.encoding)
    except UnicodeDecodeError:
        self.sendMessage(irc.ERR_NOSUCHNICK, params[0], ':No such nick/channel')
        return
    self.realm.lookupUser(user).addCallbacks(cbUser, ebUser)