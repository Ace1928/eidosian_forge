from zope.interface import Attribute, Interface
def registerAccountClient(client):
    """
        Notifies user that an account has been signed on to.

        @type client: L{Client<IClient>}
        """