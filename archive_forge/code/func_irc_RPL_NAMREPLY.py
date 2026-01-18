from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.internet.defer import succeed
from twisted.words.im import basesupport, interfaces, locals
from twisted.words.im.locals import ONLINE
from twisted.words.protocols import irc
def irc_RPL_NAMREPLY(self, prefix, params):
    """
        RPL_NAMREPLY
        >> NAMES #bnl
        << :Arlington.VA.US.Undernet.Org 353 z3p = #bnl :pSwede Dan-- SkOyg AG
        """
    group = params[2][1:].lower()
    users = params[3].split()
    for ui in range(len(users)):
        while users[ui][0] in ['@', '+']:
            users[ui] = users[ui][1:]
    if group not in self._namreplies:
        self._namreplies[group] = []
    self._namreplies[group].extend(users)
    for nickname in users:
        try:
            self._ingroups[nickname].append(group)
        except BaseException:
            self._ingroups[nickname] = [group]