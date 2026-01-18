import abc
from neutron_lib.api.definitions import portbindings
@property
@abc.abstractmethod
def original_top_bound_segment(self):
    """Return the original top-level bound segment dictionary.

        This property returns the original top-level bound segment
        dictionary, or None if the port was previously unbound. For a
        previously bound port, original_top_bound_segment is
        equivalent to original_binding_levels[0][BOUND_SEGMENT], and
        returns one of the port's network's static segments.

        This property is only valid within calls to
        update_port_precommit and update_port_postcommit. It returns
        None otherwise.
        """