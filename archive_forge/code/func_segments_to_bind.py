import abc
from neutron_lib.api.definitions import portbindings
@property
@abc.abstractmethod
def segments_to_bind(self):
    """Return the list of segments with which to bind the port.

        This property returns the list of segment dictionaries with
        which the mechanism driver may bind the port. When
        establishing a top-level binding, these will be the port's
        network's static segments. For each subsequent level, these
        will be the segments passed to continue_binding by the
        mechanism driver that bound the level above.

        This property is only valid within calls to
        MechanismDriver.bind_port. It returns None otherwise.
        """