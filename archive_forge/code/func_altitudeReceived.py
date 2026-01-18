from zope.interface import Attribute, Interface
def altitudeReceived(altitude):
    """
        Method called when an altitude is received.

        @param altitude: The altitude.
        @type altitude: L{twisted.positioning.base.Altitude}
        """