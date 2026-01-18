from zope.interface import Attribute, Interface
def headingReceived(heading):
    """
        Method called when a true heading is received.

        @param heading: The heading.
        @type heading: L{twisted.positioning.base.Heading}
        """