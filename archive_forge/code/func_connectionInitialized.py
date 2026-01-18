from zope.interface import Attribute, Interface
def connectionInitialized():
    """
        The XML stream has been initialized.

        At this point, authentication was successful, and XML stanzas can be
        exchanged over the XML stream L{xmlstream}. This method can be
        used to setup observers for incoming stanzas.
        """