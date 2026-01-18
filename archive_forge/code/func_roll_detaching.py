from cinderclient.apiclient import base as common_base
from cinderclient import base
def roll_detaching(self, volume):
    """Roll detaching this volume.

        :param volume: The :class:`Volume` (or its ID)
                       you would like to roll detaching.
        """
    return self._action('os-roll_detaching', volume)