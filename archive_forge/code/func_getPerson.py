from zope.interface import Attribute, Interface
def getPerson(name, client):
    """
        Get a Person for a client.

        Duplicates L{IAccount.getPerson}.

        @type name: string
        @type client: L{Client<IClient>}

        @rtype: L{Person<IPerson>}
        """