from twisted.application import strports
from twisted.python import usage
from twisted.words.protocols.jabber import component
def makeService(config):
    router = component.Router()
    factory = component.XMPPComponentServerFactory(router, config['secret'])
    if config['verbose']:
        factory.logTraffic = True
    return strports.service(config['port'], factory)