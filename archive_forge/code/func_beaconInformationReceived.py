from zope.interface import Attribute, Interface
def beaconInformationReceived(beaconInformation):
    """
        Method called when positioning beacon information is received.

        @param beaconInformation: The beacon information.
        @type beaconInformation: L{twisted.positioning.base.BeaconInformation}
        """